"""Wrap a scale-1 DRRN as a PETSc PC-SHELL for learned FGMRES preconditioning."""

from __future__ import annotations

import torch

try:
    from petsc4py import PETSc
except ImportError:  # PETSC_STUB
    PETSc = None  # type: ignore[assignment]

from src.bridge.petsc_torch_bridge import tensor_to_vec, vec_to_tensor


class DRRNPreconditioner:
    """Apply a native-resolution DRRN as a variable PETSc shell preconditioner.
    
    [SCIENTIFIC NOTE ON ARCHITECTURAL MISMATCH]
    This algebraic shell preconditioner approach degrades convergence because it passes 
    FGMRES residual vectors through a DRRN forward pass. A preconditioner M should 
    approximate A^-1, meaning given a residual r = b - Ax, it must map to a correction 
    delta ≈ A^-1 r. 
    However, the DRRN was trained to map coarse velocity fields to fine velocity fields 
    (smooth, O(1) velocity magnitude space), not non-smooth, tiny magnitude residual 
    vectors. Residuals and velocity fields reside in entirely different statistical spaces, 
    making this direct application numerically counterproductive.
    A correct algebraic approach requires a dedicated residual-to-correction training 
    dataset generated from FGMRES solve history — which is out of scope for the current 
    Reynolds continuation study. 
    Instead, DRRN is better suited as a two-phase outer loop spatial smoother.
    """

    def __init__(self, model: torch.nn.Module, da: "PETSc.DMDA", device: str = "cuda") -> None:
        self.model = model
        self.da = da
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def apply(self, pc: "PETSc.PC", x: "PETSc.Vec", y: "PETSc.Vec") -> None:
        """Map a PETSc residual vector through the DRRN and write the result in-place."""

        tensor = vec_to_tensor(x, device=self.device).to(dtype=torch.float32)
        with torch.no_grad():
            output = self.model(tensor)
        tensor_to_vec(output, y)


def attach_drrn_pc(ksp: "PETSc.KSP", drrn_pc: DRRNPreconditioner) -> None:
    """Replace the KSP preconditioner with the DRRN shell preconditioner."""

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(drrn_pc)
