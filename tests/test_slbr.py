import unittest
import torch

from lib.models.slbr.resunet import SLBR

class TestSLBR(unittest.TestCase):
    def test_slbr(self):
        model = SLBR(shared_depth=1, blocks=3, long_skip=False)
        x = torch.randn(8, 3, 256, 256)
        m = torch.randn(8, 1, 256, 256)
        im, mask = model(x)
        self.assertEqual(x.shape, im.shape)
        self.assertEqual(mask.shape, m.shape)


if __name__ == "__main__":
    unittest.main()
