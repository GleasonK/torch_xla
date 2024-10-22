import os
import jax
import jax.export
import tensorflow as tf
import torch
import torch.nn.functional as F
import torch_xla2
from torch_xla2 import tensor
from torch_xla2 import tf_integration
import torch_xla2.export
from torch_xla2.ops import mappings
from google3.testing.pybase import googletest as unittest


class Interpolate(torch.nn.Module):

  def forward(self, masks: torch.Tensor) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        size=(500, 500),
        mode="bilinear",
        align_corners=False,
    )
    return masks


class TensorConstant(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, a):
    return a / torch.tensor(3)


class TfIntegrationTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)

  def test_interpolate(self):
    """Simple model roundtripped through TF savedmodel"""

    # Check Accuracy
    arg = (torch.randn(3, 3, 200, 200),)
    pt_model = Interpolate()

    sm_path = os.path.join(self.create_tempdir(), "interpolate.savedmodel")
    tf_model = tf_integration.save_torch_module_as_tf_saved_model(
        pt_model, arg, sm_path)
    print(tf_model)
    loaded_model = tf.saved_model.load(sm_path)

    pt_res = pt_model(*arg)
    tf_res = torch.tensor(loaded_model.f(*arg)[0].numpy())
    print(pt_res)
    print(tf_res)
    self.assertTrue(torch.allclose(pt_res, tf_res, atol=1e-4))


if __name__ == "__main__":
  unittest.main()
