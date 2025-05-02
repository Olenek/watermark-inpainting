import unittest
import torch

from lib.models.rirci.glci.base import scSEBlock, SpectralTransform, LocalMLP, GlobalMLP
from lib.models.rirci.glci.glci import GLCI, GLCIStack


class TestGLCISubmodules(unittest.TestCase):
    def test_scse(self):
        scse = scSEBlock(16)
        x = torch.randn(1, 16, 32, 32)
        y = scse(x)
        self.assertEqual(y.shape, x.shape)

    def test_spectral_transform(self):
        spectral = SpectralTransform()
        x = torch.randn(1, 16, 32, 32)
        y = spectral(x)
        self.assertEqual(y.shape, x.shape)

    def test_local_mlp(self):
        local_mlp = LocalMLP(16)
        x = torch.randn(1, 16, 32, 32)
        y = local_mlp(x)
        self.assertEqual(y.shape, x.shape)

    def test_global_mlp(self):
        global_mlp = GlobalMLP(16)
        x = torch.randn(1, 16, 32, 32)
        y = global_mlp(x)
        self.assertEqual(y.shape, x.shape)

    def test_glci(self):
        glci = GLCI(in_channels=4)
        x = torch.randn(1, 4, 32, 32)
        y = glci(x)
        self.assertEqual(y.shape, x.shape)

    def test_glci_stack(self):
        glci_stack = GLCIStack(in_channels=4, n_blocks=3)
        x = torch.randn(1, 4, 32, 32)
        y = glci_stack(x)
        self.assertEqual(y.shape, x.shape)

if __name__ == "__main__":
    unittest.main()
