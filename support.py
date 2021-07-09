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
        self.residual = nn.ModuleList(
            [
               Residual(n_atom_basis, activation) 
               for _ in range(n_interactions)
            ]
        )
        self.sequential_i = nn.Sequential(
                    activation(),
                    nn.Linear(n_atom_basis, n_atom_basis),
                    activation()
        )
        self.sequential_j = nn.Sequential(
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
        
        x_j = x[idx_j]
        xp = torch.rand(self.n_atom_basis)*x ## check this
        vp = self.sequential_j(x_j.float())*self.linear_g(g_ij.float())
        vm = self.sequential_i(x)
        v = snn.scatter_add(vp, idx_i, dim_size=n_atoms) + vm
        for module in self.residual:
            v = module(v)
        
        v = self.activation(v)
        
        return xp + self.linear_f(v)
    
class PhysNetCutOff(nn.Module):
    
    def __init__(self, cutoff: float):
        super(PhysNetCutOff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, d_ij: torch.Tensor):


        # Compute values of cutoff function
        input_cut = 1 - 6*(d_ij/self.cutoff)**5 + 15*(d_ij/self.cutoff)**4 - 10*(d_ij/self.cutoff)**3
        # Remove contributions beyond the cutoff radius
        input_cut *= (d_ij < self.cutoff).float()
        return input_cut