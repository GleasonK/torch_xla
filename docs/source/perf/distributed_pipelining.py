#!/usr/bin/env python
# coding: utf-8

# # Pipeline Parallelism
# 
# ## Simple Pipeline
# 
# No local SPMD or FSDP.

# In[16]:


import os
os.environ["WORLD_SIZE"] = '8'
os.environ["CPU_NUM_DEVICES"] = os.environ["WORLD_SIZE"]
os.environ["PJRT_DEVICE"] = 'CPU'


# In[17]:


from typing import Optional

import numpy as np
import torch
from torch import nn

class SimpleLinear(nn.Module):
  NUM_CLASSES = 3

  def __init__(self, input_dim):
      super().__init__()
      # Instead of Sequential, define layers separately for easier split points
      self.layer0 = nn.Linear(input_dim, input_dim // 2)
      self.relu = nn.ReLU()
      self.layer1 = nn.Linear(input_dim // 2, 3)
      self.layer2 = nn.Linear(3, self.NUM_CLASSES)

  def forward(self, x):
      x = self.layer0(x)
      x = self.relu(x)
      x = self.layer1(x)
      x = self.layer2(x)
      return x


# ## Simple Pipeline - No FSDP or TP

# In[41]:


from torch.distributed.pipelining import ScheduleGPipe, SplitPoint, pipeline, PipelineStage

world_size = int(os.environ["WORLD_SIZE"])

class TrainingOptions():
    def __init__(self):
      self.batch_size = 128
      self.num_epochs = 1
      self.lr = 0.1
      self.log_steps = 8
      self.input_dim = 16834
      self.train_dataset_len = 1024 * 8
      self.pipeline_chunks = 2

opts = TrainingOptions()

model = SimpleLinear(opts.input_dim)

# Define split points for pipeline parallelism
split_spec = {
  "layer0": SplitPoint.END,
}

# Create a sample input for the pipeline
chunks = opts.pipeline_chunks
batch_size = opts.batch_size

example_input = torch.randn(batch_size, opts.input_dim)

# Make sure that program is full-graph capturable:
# torch.export.export(model, (example_input,))

# Create the pipeline, returns a GraphModule of submodule calls
pipe = pipeline(model, mb_args=(example_input,), split_spec=split_spec)


# # Exporting Pipeline via JaxInterpreter
# Note that this pipeline contains all the individual stages of interest.
# Given that this was formed from `torch.export` it seems plausible that we can
# get each sub-module asa a separately traced StableHLO program to be stitched
# together.
# 
# Impl ideas:
# 
# 1. Trace each subprogram separately, then stub out call_module of GraphModule
#    into a custom MPMD-like PT op that doesn't actually call into the submodule.
# 2. Overload call_module to produce a custom MPMD call op _and_ trace the
#    subprogram as its own function. (Hitting issues I believe that are
#    decomposition related).

# ## PyTorch Pipeline to MPMD Program FX Interpreter
# 
# This is a helper class that can live in one of our libraries to export MPMD
# programs to XLA. An example of its use is in the following section.

# In[44]:


import jax
import torchax
from torchax.export import JaxInterpreter
from typing import Any, Dict, Tuple, Optional
import functools
from torch.utils import _pytree as pytree

DEBUG_LEVEL=1

def dprint(lvl, str):
  if lvl <= DEBUG_LEVEL:
    print(str)

class PipelineInterpreter(JaxInterpreter):
  """A subclass of the Torch to JAX exporter for Pipelines
  The main difference being that pipelines can call submodules and need to
  manage devices for each submodule.
  """
  def __init__(self, pipe, global_mesh, **kwargs):
    if not isinstance(pipe, torch.distributed.pipelining.Pipe):
      raise ValueError(f"Input arg {type(pipe)} is not a torch Pipe or GraphModule")
    self.mesh = global_mesh
    self.submodule_mesh_map = {}
    
    # Map from Stage to Local SPMD world
    stage_names = [n[0] for n in pipe.split_gm.named_modules() if isinstance(n[1], torch.fx.GraphModule)][1:]
    assert len(stage_names) == len(global_mesh.get_logical_mesh()), "Global mesh size much match num stages"
    for i in range(len(stage_names)):
      self.submodule_mesh_map[stage_names[i]] = global_mesh.get_logical_mesh()[i]

    # Init interpreter using the parent graph
    dprint(1, f"Init using mesh {global_mesh}")
    dprint(2, f"Init using pipe {pipe}")
    dprint(1, f"Init using mesh map {self.submodule_mesh_map}")
    super().__init__(pipe.split_gm, **kwargs)

  def to_example_input(self, tensor):
    """Create example tensors for exporting pipeline stages"""
    dprint(1, f"Converting: {tensor}")
    shape = tensor.shape
    dtype = tensor.dtype
    if not isinstance(dtype, torch.dtype):
      dtype = torchax.tensor.j2t_dtype(dtype)
    return torch.rand(shape, dtype=dtype)

  def call_module(self,
                  target: str,
                  args: Optional[tuple["Argument", ...]] = None,
                  kwargs: Optional[dict[str, "Argument"]] = None) -> Any:
    dprint(1, f"call_module {target}, {args}, {kwargs}")
    
    # Get the submodule
    submodule = self.fetch_attr(target)
    dprint(2, submodule)

    if not isinstance(submodule, torch.fx.GraphModule):
      print("NOT GRAPH MODULE. Probably should never be reached?")
      return super().call_module(target, args=args, kwargs=kwargs)
  
    # Trace GraphModule->Jaxpr / MLIR Module
    args, _ = pytree.tree_flatten(args)
    ep_args = tuple(self.to_example_input(arg) for arg in args)
    # TODO: Handle kwargs?
    ep = torch.export.export(submodule, args=ep_args, kwargs=kwargs)
    weights, func = torchax.export.exported_program_to_jax(ep)

    # JIT and Call - Note this outlines the subprogram as a function, not a
    # separate module
    jax_args, _ = pytree.tree_flatten((args, kwargs))
    dprint(2, jax.jit(func).lower(weights, jax_args)._lowering.stablehlo())

    def mpmd_stage(*args, **kwargs):
      del kwargs
      return func(*args)

    devices = self.submodule_mesh_map[target]
    return jax.lax.composite(mpmd_stage, name="mpmd.call")(weights, jax_args, devices=devices)

  def run_node(self, n) -> Any:
    dprint(1, f"Pipeline interpreter running node {n}")
    return super().run_node(n)


# Export the pipeline to a monolithic StableHLO module with orchestration via
# custom "MPMD" ops that will be better standardized in the future.
def export_pipeline(pipe, global_mesh):
  def wrapper(*args):
    return PipelineInterpreter(pipe, global_mesh).run(*args, enable_io_processing=False)
  return jax.jit(xla_pipeline).lower(torchax.tensor.t2j(example_input))._lowering.stablehlo()


# Use the MPMD Export function `export_pipeline` defined above.

# In[45]:


# Create a global mesh of all available devices
# Note there must be 1 dimension per chunk (is this a fair assumption?)
# Also in this instance we are only sharding on data, model parallel = 1.
import torch_xla.distributed.spmd as xs
chunks = pipe.num_stages
tp = world_size // chunks
mp = 1

global_mesh_shape = (chunks, tp, mp)
global_mesh = xs.Mesh(np.arange(world_size), global_mesh_shape, ("pp", "data", "model"))

module = export_pipeline(pipe, global_mesh)

print("\n=== MPMD Module Output ===\n")
module.operation.print(large_elements_limit=100)


# ## The Missing Pieces
# 
# A few thoughts of what's incomplete / not fully thought through:
# 
# - More standardized MPMD ops, probably something that is registered to torch
#   this may allow us to control our lowerings to a HW specific desired
#   representation?
# - A runtime for the above "monolithic mpmd program" needs to exist.
#   + We roughly have something internal, lowers this to IFRT (public dialect) and
#     build MPMD runtime on top of IFRT operations.
# - We still need Local SPMD to load data into the subset of devices interacting
#   the monoprogram at the top level.
# - Need to figure out how to do TP + FSDP in coordination with this approach,
#   in theory if TP/FSDP are possible in `torch.export` we can leverage that.
#   + How does a training loop with backward passes work with this? Do it at the
#     pre-pipelined module level? Or on the exported pipeline?
