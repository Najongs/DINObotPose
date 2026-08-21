"""Rigid alignment helper shared by the training scripts.

Extracted verbatim from the research tree so the release does not have to ship the
diagnostic script it lived in.
"""
import torch


def kabsch_batch(A, B):
    ca = A.mean(1, keepdim=True); cb = B.mean(1, keepdim=True)
    H = torch.einsum('bni,bnj->bij', A - ca, B - cb)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.det(torch.einsum('bij,bjk->bik', Vt.transpose(1, 2), U.transpose(1, 2)))
    D = torch.eye(3, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1); D[:, 2, 2] = d
    R = torch.einsum('bij,bjk,bkl->bil', Vt.transpose(1, 2), D, U.transpose(1, 2))
    t = (cb - torch.einsum('bij,bnj->bni', R, ca)).squeeze(1)
    return R, t
