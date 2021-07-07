import torch
import torch.nn as nn
from support import Residual, Output

import schnetpack.properties as structure
import schnetpack.nn as snn

from typing import Callable, Dict


class PhysNet(nn.Module):

    def __init__(
        self,
        n_atom_basis:int,
        activation: Callable,
        radial_basis : Callable,
        cutoff_fn: Callable,
        n_rbf:int,
        n_interactions:int,
        n_output_residual:int,
        n_atomic_residual:int,
        n_modules: int,
        max_z = 200
    ):
        super(PhysNet, self).__init__()
        self.rbf = radial_basis
        self.cutoff_fn = cutoff_fn
        self.radial_basis = radial_basis
        
        self.module = nn.ModuleList(
            [Module(n_atom_basis,
                             activation,
                             n_rbf,
                             n_interactions,
                             n_output_residual,
                             n_atomic_residual)
             for _ in range(n_modules)
            ]
        )
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)
        
    def forward(self,inputs):
        
        atomic_numbers = inputs[structure.Z]
        r_ij = inputs[structure.Rij]
        idx_i = inputs[structure.idx_i]
        idx_j = inputs[structure.idx_j]
        n_atoms = atomic_numbers.shape[0]
        d_ij = torch.norm(r_ij, dim=1, keepdim=True).squeeze(-1)
        
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)
        g_ij = torch.einsum('ij,i->ij', phi_ij, fcut) 
        x = self.embedding(atomic_numbers)
        
        summation = torch.zeros(x.shape)
        
        for module in self.module:
            xo, x = module(x, g_ij, idx_i, idx_j, n_atoms)
            summation = summation + xo 
        
   
        return summation