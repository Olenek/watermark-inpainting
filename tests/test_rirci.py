import torch

from lib.rirci.exclusion import RIRCIStage1
from lib.rirci.restoration import RIRCIStage2


def test_rirci_stage1():
    """
    Tests full forward pass of RIRCIStage1 on dummy input.
    Ensures proper output shapes and valid mask values.
    """
    model = RIRCIStage1()
    model.eval()

    B, C, H, W = 2, 3, 256, 256
    J = torch.randn(B, C, H, W)

    with torch.no_grad():
        M_hat, C_b_hat = model(J)

    # Check mask output
    assert M_hat.shape == (B, 1, H, W), f"Mask shape mismatch: {M_hat.shape}"
    assert torch.all(M_hat >= 0) and torch.all(M_hat <= 1), "Mask values not in [0, 1] range"

    # Check background component output
    assert C_b_hat.shape == (B, 3, H, W), f"Background component shape mismatch: {C_b_hat.shape}"

    print("✅ RIRCIStage1 passed")

def test_rirci_stage2():
    model = RIRCIStage2()
    model.eval()

    B, H, W = 2, 256, 256
    J = torch.randn(B, 3, H, W)
    M_hat = torch.randn(B, 1, H, W)
    C_b = torch.randn(B, 3, H, W)

    with torch.no_grad():
        I_r, I_i, I_hat = model(J, M_hat, C_b)

    # Check mask output
    assert I_hat.shape == (B, 3, H, W)
    assert I_r.shape == I_i.shape == I_hat.shape, f"Image shape mismatch: {I_r.shape, I_i.shape, I_hat.shape}"

    print("✅ RIRCIStage2 passed")


if __name__ == '__main__':
    test_rirci_stage2()
