import unittest
import torch

from lib.loss import MaskedL1Loss, VGGPerceptualLoss


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.channels = 3
        self.height = self.width = 256  # Match your model's expected input size

    # Tests will go here

    def test_masked_l1_forward_pass(self):
        """Test forward pass with valid inputs"""
        loss_fn = MaskedL1Loss().to(self.device)

        # Synthetic data
        X = torch.rand(self.batch_size, self.channels, self.height, self.width,
                       device=self.device)
        Y = torch.rand_like(X)
        M = torch.randint(0, 2, (self.batch_size, 1, self.height, self.width),
                          device=self.device).float()

        loss = loss_fn(X, Y, M)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))  # Scalar output

    def test_masked_l1_gradient_flow(self):
        """Test gradients propagate correctly"""
        loss_fn = MaskedL1Loss().to(self.device)
        X = torch.rand(1, 3, 64, 64, device=self.device, requires_grad=True)
        Y = torch.rand_like(X)
        M = torch.ones_like(X[:, :1])  # All pixels masked

        loss = loss_fn(X, Y, M)
        loss.backward()
        self.assertIsNotNone(X.grad)  # Gradients should exist

    def test_masked_l1_empty_mask(self):
        """Test when mask is all zeros (no active region)"""
        loss_fn = MaskedL1Loss().to(self.device)
        X = torch.ones(1, 3, 64, 64, device=self.device)
        Y = torch.zeros_like(X)
        M = torch.zeros_like(X[:, :1])  # No masked region

        loss = loss_fn(X, Y, M)
        self.assertEqual(loss.item(), 0.0)  # Expect zero loss

    def test_masked_l1_channel_mismatch(self):
        """Test automatic mask channel expansion"""
        loss_fn = MaskedL1Loss().to(self.device)
        X = torch.rand(1, 3, 64, 64, device=self.device)
        Y = torch.rand_like(X)
        M = torch.randint(0, 2, (1, 1, 64, 64), device=self.device).float()

        # Should work despite mask having 1 channel vs. image's 3
        loss = loss_fn(X, Y, M)
        self.assertTrue(loss.item() >= 0)

    def test_vgg_perceptual_forward_pass(self):
        """Test basic forward pass"""
        loss_fn = VGGPerceptualLoss().to(self.device)
        X = torch.rand(self.batch_size, self.channels, self.height, self.width,
                       device=self.device) * 2 - 1  # [-1, 1] range
        Y = torch.rand_like(X)

        loss = loss_fn(X, Y)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))  # Scalar output
        self.assertTrue(loss.item() >= 0)

    def test_vgg_perceptual_identical_inputs(self):
        """Test loss=0 when inputs are identical"""
        loss_fn = VGGPerceptualLoss().to(self.device)
        X = torch.rand(1, 3, 256, 256, device=self.device) * 2 - 1
        loss = loss_fn(X, X)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)  # Allow small numerical errors

    def test_vgg_perceptual_gradient_flow(self):
        loss_fn = VGGPerceptualLoss().to(self.device)
        X = (torch.rand(1, 3, 256, 256, device=self.device) * 2 - 1).detach().requires_grad_()
        Y = torch.rand_like(X)

        loss = loss_fn(X, Y)
        loss.backward()

        self.assertIsNotNone(X.grad)
        self.assertFalse(torch.all(X.grad == 0))  # Gradients should be non-zero

    def test_vgg_perceptual_input_range(self):
        """Test handling of out-of-range inputs"""
        loss_fn = VGGPerceptualLoss().to(self.device)
        X = torch.rand(1, 3, 256, 256, device=self.device)  # [0,1] range (invalid)
        Y = torch.rand_like(X)

        # Should handle normalization internally
        loss = loss_fn(X, Y)
        self.assertTrue(loss.item() >= 0)

    # def test_masked_l1_nan_handling(self):
    #     """Test NaN input handling"""
    #     loss_fn = MaskedL1Loss().to(self.device)
    #     X = torch.rand(1, 3, 64, 64, device=self.device)
    #     Y = torch.rand_like(X)
    #     M = torch.ones_like(X[:, :1])
    #     X[0, 0, 0, 0] = float('nan')  # Introduce NaN
    #
    #     loss = loss_fn(X, Y, M)
    #     self.assertFalse(torch.isnan(loss))

    def test_vgg_perceptual_different_sizes(self):
        """Test non-square input handling"""
        loss_fn = VGGPerceptualLoss().to(self.device)
        X = torch.rand(1, 3, 224, 256, device=self.device) * 2 - 1
        Y = torch.rand_like(X)

        loss = loss_fn(X, Y)
        self.assertTrue(loss.item() >= 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
