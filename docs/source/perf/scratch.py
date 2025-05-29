import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs

# Use JAX for the MLIR python bindings
from jax._src.interpreters import mlir
from jax._src.lib.mlir import dialects, ir, passmanager
from jaxlib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
from jax._src.lib.mlir.dialects import func as func_dialect

def build_example_input_from_result(module):
  main = [op for op in module.body.operations if isinstance(op, func_dialect.FuncOp)][0]
  result = main.body.blocks[0].operations[-1].operation.operands[0].type
  shape = tuple(result.shape)
  dtype = result.element_type
  example_input = torch.rand(shape, dtype=torch.float32)
  return example_input

def export_pipeline_stage(pipe, idx, stage_names, example_input, global_mesh, ctx):
  # Build the local mesh for the stage (pp, tp, dp)
  local_device_ids = global_mesh.get_logical_mesh()[0] # FIXME: [idx] once local shapes allowed
  local_mesh_shape = (global_mesh.shape()['data'], global_mesh.shape()['model'])
  local_mesh = xs.Mesh(local_device_ids, local_mesh_shape, ("data", "model"))
  print("Local Mesh:", local_mesh)

  # Annotate the stage input with proper sharding info
  xs.mark_sharding(example_input, mesh=local_mesh, partition_spec=("data","model"))
  
  # Get the pipeline stage
  stage = pipe.get_stage_module(idx)
  stage_name = stage_names[idx]
  # print("Stage", stage)

  # Export stage to StableHLO
  exported = torch.export.export(stage, args=(example_input,))
  # print("Exported", exported)
  bytecode = torch_xla.stablehlo.exported_program_to_stablehlo(exported).get_stablehlo_bytecode()
  module = stablehlo.deserialize_portable_artifact(ctx, bytecode)
  module.operation.attributes['sym_name'] = ir.StringAttr.get(stage_name)

  # Build example input for next stage
  next_inputs = build_example_input_from_result(module)
  return module, next_inputs

def pipeline_to_gmpmd(pipe, example_input):
  xr.use_spmd()
  num_devices = xr.global_runtime_device_count()

  # Create Global Mesh
  chunks = 1 # FIXME: pipe.num_stages
  tp = num_devices // chunks
  mp = 1

  global_mesh_shape = (chunks, tp, mp)
  global_mesh = xs.Mesh(np.arange(num_devices), global_mesh_shape, ("pp", "data", "model"))

  stage_names = [n[0] for n in pipe.named_modules() if isinstance(n[1], torch.fx.GraphModule)][1:]
  with mlir.make_ir_context() as ctx:
    submodules = []
    for stage_idx in range(pipe.num_stages):
      print("Exporting stage", stage_idx)
      example_input = example_input.to('xla')
      (stage_ir, example_input) = export_pipeline_stage(pipe, stage_idx, stage_names, example_input, global_mesh, ctx)
      submodules.append(stage_ir.operation)

    for submodule in submodules:
      print(submodule)
    # mpmd_module = ir.Module.create(loc=ir.Location.unknown())
    # sym_tab = ir.SymbolTable(mpmd_module.operation)
    # for submodule in submodules:
    #   mpmd_module.body.append(submodule.operation)
    # for submodule in submodules:
      # print(submodule)
      # sym_tab.insert(stage_ir)
    # print(mpmd_module)
    
print(pipe)
pipeline_to_gmpmd(pipe, example_input)