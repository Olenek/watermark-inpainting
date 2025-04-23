import unittest
import torch

from lib.rirci.resunet import SLBR, CoarseEncoder, SharedBottleNeck, CoarseDecoder


class TestGLCISubmodules(unittest.TestCase):
    def test_encoder_decoder(self):
        in_channels = 3
        depth = 5
        blocks = 1
        start_filters = 32
        shared_depth = 2

        encoder = CoarseEncoder(in_channels=in_channels, depth=depth - shared_depth, blocks=5,
                                     start_filters=start_filters)
        shared_decoder = SharedBottleNeck(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                               depth=depth, shared_depth=shared_depth, blocks=1)

        x = torch.randn(8, 3, 256, 256)
        m = torch.randn(8, 1, 256, 256)
        image_code, before_pool = encoder(x)

        im, mask = shared_decoder(image_code)
        self.assertEqual(x.shape, im.shape)
        self.assertEqual(mask.shape, m.shape)

    def test_slbr(self):
        model = SLBR()
        x = torch.randn(8, 3, 256, 256)
        m = torch.randn(8, 1, 256, 256)
        im, mask = model(x)
        self.assertEqual(x.shape, im.shape)
        self.assertEqual(mask.shape, m.shape)


if __name__ == "__main__":
    unittest.main()
