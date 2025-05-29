import os
os.environ["WORLD_SIZE"] = '8'
os.environ["RANK"] = '0'
os.environ["CPU_NUM_DEVICES"] = os.environ["WORLD_SIZE"]
os.environ["PJRT_DEVICE"] = 'CPU'

import torch_xla
torch_xla.runtime.global_runtime_device_attributes()

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

from torch.distributed.pipelining import ScheduleGPipe, SplitPoint, pipeline, PipelineStage

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])

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
device = 'xla'

model = SimpleLinear(opts.input_dim).to(device)

# Define split points for pipeline parallelism
split_spec = {
  "layer0": SplitPoint.END,
}

# Create a sample input for the pipeline
chunks = opts.pipeline_chunks
batch_size = opts.batch_size

example_input = torch.randn(batch_size, opts.input_dim, device=device)
pipe = pipeline(model, mb_args=(example_input,), split_spec=split_spec)
print(pipe)

import jax
import torchax
from torchax.export import JaxInterpreter
from typing import Any, Dict, Tuple, Optional

class PipelineInterpreter(JaxInterpreter):
  """A subclass of the Torch to JAX exporter for Pipelines
  The main difference being that pipelines can call submodules and need to
  manage devices for each submodule.
  """

  def __init__(self, pipe, global_mesh):
    if not isinstance(pipe, torch.distriuted.Pipe):
      raise ValueError(f"Input arg {pipe} is not a torch.distributed.Pipe")
    super().__init__(pipe.split_gm.to('jax'))

  def call_module(self,
                  target: str,
                  args: Optional[tuple["Argument", ...]] = None,
                  kwargs: Optional[dict[str, "Argument"]] = None) -> Any:
    print("call_module", target, args, kwargs)
    submodule = self.fetch_attr(target)
    if (isinstance(submodule, torch.fx.GraphModule)):
      print("Tracing graphmodule")
      return JaxInterpreter(submodule).run(*args, **kwargs)
    print("Super")
    return super().call_module(target, args=args, kwargs=kwargs)
  
  def run_node(self, n) -> Any:
    print("run node", n)
    return super().run_node(n)

example_input = torch.randn(batch_size, opts.input_dim, device=device)
def foo(example_input, mesh):
  return PipelineInterpreter(pipe, mesh).run(example_input)

print(jax.jit(foo).lower(example_input)._lowering.stablehlo())