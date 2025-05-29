#!/usr/bin/env python
# coding: utf-8

# # Pipeline Parallelism
# 
# ## Simple Pipeline
# 
# No local SPMD or FSDP.

# In[ ]:


import os
os.environ["WORLD_SIZE"] = '8'
# os.environ["CPU_NUM_DEVICES"] = os.environ["WORLD_SIZE"]
os.environ["PJRT_DEVICE"] = 'CPU'


# In[ ]:


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

# In[ ]:


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
with torch.no_grad():
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

# In[ ]:


import jax
import torchax
from torchax.export import JaxInterpreter
from typing import Any, Dict, Tuple, Optional
import functools
from torch.utils import _pytree as pytree

DEBUG_LEVEL=0

def printD(lvl, str):
  if lvl >= DEBUG_LEVEL:
    print(str)

class PipelineInterpreter(JaxInterpreter):
  """A subclass of the Torch to JAX exporter for Pipelines
  The main difference being that pipelines can call submodules and need to
  manage devices for each submodule.
  """

  def to_example_input(self, tensor):
    printD(1, f"Converting: {tensor}")
    shape = tensor.shape
    dtype = tensor.dtype
    if not isinstance(dtype, torch.dtype):
      dtype = torchax.tensor.j2t_dtype(dtype)
    return torch.rand(shape, dtype=dtype)

  def __init__(self, pipe, global_mesh, **kwargs):
    if not isinstance(pipe, torch.distributed.pipelining.Pipe):
      raise ValueError(f"Input arg {type(pipe)} is not a torch Pipe or GraphModule")
    self.mesh = global_mesh
    print("INIT")
    super().__init__(pipe.split_gm, **kwargs)

  def call_module(self,
                  target: str,
                  args: Optional[tuple["Argument", ...]] = None,
                  kwargs: Optional[dict[str, "Argument"]] = None) -> Any:
    printD(1, f"call_module {target}, {args}, {kwargs}")
    submodule = self.fetch_attr(target)
    printD(2, submodule)

    if not isinstance(submodule, torch.fx.GraphModule):
      print("not graphmodule, probably shouldn't be hit?")
      return super().call_module(target, args=args, kwargs=kwargs)
  
    # Trace GraphModule->Jaxpr / MLIR Module
    args, _ = pytree.tree_flatten(args)
    ep_args = tuple(self.to_example_input(arg) for arg in args)
    # TODO: Handle kwargs
    ep = torch.export.export(submodule, args=ep_args, kwargs=kwargs)
    weights, func = torchax.export.exported_program_to_jax(ep)

    # JIT and Call - Note this outlines the subprogram as a function, not a
    # separate module
    jax_args, _ = pytree.tree_flatten((args, kwargs))
    printD(2, jax.jit(func).lower(weights, jax_args)._lowering.stablehlo())
    return jax.jit(func)(weights, jax_args)

  def run_node(self, n) -> Any:
    printD(1, f"Pipeline interpreter running node {n}")
    return super().run_node(n)


def foo(example_input):
  return PipelineInterpreter(pipe, None).run(example_input, enable_io_processing=False)

module = jax.jit(foo).lower(torchax.tensor.t2j(example_input))._lowering.stablehlo()
module.operation.print(large_elements_limit=100)

