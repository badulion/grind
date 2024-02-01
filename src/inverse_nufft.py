import torch.nn as nn
import torch    
    
class FourierInterpolator(nn.Module):
    def __init__(self, num_ks = 5, spatial_dim: int = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_ks = num_ks
        self.spatial_dim = spatial_dim
        
    def forward(self, points_source, values_source, points_target):
        """ approximates the function at the given points using the fourier coefficients """
        fourier_coefficients = self.solve_for_fourier_coefficients(points_source, values_source)
        basis = self.generate_fourier_basis(points_target)
        
        reconstruction = (basis @ fourier_coefficients).real
        return reconstruction
    
    def generate_fourier_basis(self, points):
        points = 2*torch.pi*(points-0.5)
        ks = self.generate_fourier_ks(points)
        return torch.exp(1j * (points @ ks.T))
    
    def generate_fourier_ks(self, points):
        if self.num_ks % 2 == 0:
            ks = torch.arange(-self.num_ks//2, self.num_ks//2, dtype=points.dtype, device=points.device)
        else:
            ks = torch.arange(-(self.num_ks-1)//2, (self.num_ks-1)//2+1, dtype=points.dtype, device=points.device)
            
        ks_ = torch.meshgrid(*[ks]*self.spatial_dim)
        ks = torch.stack([k.flatten() for k in ks_], axis=1)
        return ks
    
    def solve_for_fourier_coefficients(self, points, values):
        basis = self.generate_fourier_basis(points)
        coeffs = torch.linalg.lstsq(basis, values+0j)[0]
        return coeffs