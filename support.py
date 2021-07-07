import torch
import torch.nn as nn
import schnetpack.nn as snn

from typing import Callable, Dict

class Residual(nn.Module):
    
    def __init__(
        self,
        n_atom_basis:int,
        activation: Callable, 
    ):
        super(Residual, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.activation = activation
        self.sequential = nn.Sequential(
            activation(),
            nn.Linear(n_atom_basis, n_atom_basis),
            activation(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )
        
    def forward(self,x):
        return self.sequential(x) + x    
    
    
class Output(nn.Module):
    
    def __init__(
        self,
        n_atom_basis:int,
        activation: Callable,
        n_output_residual:int
    ):
        super(Output, self).__init__()
        self.activation = activation()
        self.linear = nn.Linear(n_atom_basis, n_atom_basis)
        self.residual = nn.ModuleList(
            [
               Residual(n_atom_basis, activation) 
               for _ in range(n_output_residual)
            ]
        )
        
    def forward(self,x):
        
        for module in self.residual:
            x = module(x)
        x = self.activation(x)
        
        return self.linear(x)
    
class Module(nn.Module):
    
    def __init__(
        self,
        n_atom_basis:int,
        activation: Callable,
        n_rbf:int,
        n_interactions:int,
        n_output_residual:int,
        n_atomic_residual:int
    ):
        super(Module, self).__init__()
        self.activation = activation()
        self.linear = nn.Linear(n_atom_basis, n_atom_basis)
        self.residual = nn.ModuleList(
            [
               Residual(n_atom_basis, activation) 
               for _ in range(n_atomic_residual)
            ]
        )
        self.interaction= PhysNetInteraction(n_atom_basis, 
                                             activation, 
                                             n_rbf, 
                                             n_interactions)
        self.output = Output(n_atom_basis, 
                             activation, 
                             n_output_residual)
        
    def forward(
        self,
        x: torch.Tensor,
        g_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        n_atoms: int
    ):
        
        x = self.interaction(x, g_ij, idx_i, idx_j, n_atoms)
        for module in self.residual:
            x = module(x)
        return self.output(x), x
    
class PhysNetInteraction(nn.Module):

    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        n_rbf: int,
        n_interactions: int
    ):

        super(PhysNetInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.n_rbf = n_rbf
        self.activation = activation()
        self.n_interactions = n_interactions
        
        self.linear_f = nn.Linear(n_atom_basis, n_atom_basis) 
        self.linear_g = nn.Linear(n_rbf, n_atom_basis) 
        self.linear_i = nn.Linear(n_atom_basis, n_atom_basis)
        self.linear_j = nn.Linear(n_atom_basis, n_atom_basis)
        self.residual = nn.ModuleList(
            [
               Residual(n_atom_basis, activation) 
               for _ in range(n_interactions)
            ]
        )
        self.sequential = nn.Sequential(
                    activation(),
                    nn.Linear(n_atom_basis, n_atom_basis),
                    activation()
        )

    def forward(
        self,
        x: torch.Tensor,
        g_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        n_atoms: int
    ):
        
        x_i = x[idx_i]
        x_j = x[idx_j]
        x = torch.rand(self.n_atom_basis)*x ## check this
        
        vp = self.sequential(x_j.float())*self.linear_g(g_ij.float())
        vm = self.sequential(x_i)
        v = snn.scatter_add(vm, idx_i, dim_size=n_atoms)
        for module in self.residual:
            v = module(v)
        
        v = self.activation(v)
        
        return x + self.linear_f(v)