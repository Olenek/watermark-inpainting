import torch

from lib.models.rirci.base import DownBlock, UpBlock, SMRBranch, MBEBranch, ConvBlock


def test_conv_block():
    block = ConvBlock(in_channels=16, out_channels=32)
    x = torch.randn(1, 16, 64, 64)
    y = block(x)
    assert y.shape == (1, 32, 64, 64), f"ConvBlock output shape mismatch: {y.shape}"
    print("✅ ConvBlock passed")

def test_down_block():
    block = DownBlock(in_channels=16, out_channels=32)
    x = torch.randn(1, 16, 64, 64)
    y = block(x)
    assert y.shape == (1, 32, 32, 32), f"DownBlock output shape mismatch: {y.shape}"
    print("✅ DownBlock passed")


def test_up_block():
    # UpBlock(up_channels, out_channels)
    block = UpBlock(in_channels=64, out_channels=32)

    # upsampled input = (B, 64, 32, 32) -> becomes (B, 32, 64, 64)
    x = torch.randn(1, 64, 32, 32)

    # skip connection = (B, 64, 64, 64)
    skip = torch.randn(1, 32, 64, 64)

    y = block(x, skip)
    assert y.shape == (1, 32, 64, 64), f"UpBlock output shape mismatch: {y.shape}"
    print("✅ UpBlock passed")


def test_smr_branch():
    """Test SMR output shape and value range"""
    smr = SMRBranch(in_channels=64)
    x = torch.randn(1, 64, 256, 256)
    mask = smr(x)

    assert mask.shape == (1, 1, 256, 256), f"SMR output shape mismatch: {mask.shape}"
    assert torch.all(mask >= 0) and torch.all(mask <= 1), "SMR mask not in [0,1] range"
    print("✅ SMRBranch passed")

def test_mbe_branch():
    """Test MBE output shape"""
    mbe = MBEBranch(in_channels=64)
    x = torch.randn(1, 64, 256, 256)
    c_w = mbe(x)

    assert c_w.shape == (1, 3, 256, 256), f"MBE output shape mismatch: {c_w.shape}"
    print("✅ MBEBranch passed")

def run_all_tests():
    test_conv_block()
    test_down_block()
    test_up_block()
    test_smr_branch()
    test_mbe_branch()

if __name__ == "__main__":
    run_all_tests()