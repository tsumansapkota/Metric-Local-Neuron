import torch
import torch.nn as nn
from typing import Union, Tuple
import numpy as np


### Activation output=1 for input=1 with shift-scale
class OneActiv(nn.Module):
    '''
    Mode:
    -softplus : default
    -relu
    -exp_1.6
    -exp_abs
    '''
    def __init__(self, input_dim, mode='softplus', beta_init=0):
        super().__init__()
        self.input_dim = input_dim
        self.beta = nn.Parameter(torch.ones(1, input_dim)*beta_init)
        self.func_mode = None
        if mode == "softplus":
            self.func_mode = self.func_softplus
        elif mode == "exp_1.6":
            self.func_mode = self.func_exp_16
        elif mode == "exp_abs":
            self.func_mode = self.func_exp_abs
        elif mode == 'relu':
            self.func_mode = self.func_relu
        else:
            raise ValueError(f"mode: {mode} not recognized")
        pass
        
    def func_softplus(self, x):
        x = torch.exp(self.beta)*(x-1) + 1
        x = nn.functional.softplus(x, beta=6)
        return x
    
    def func_relu(self, x):
        x = torch.exp(self.beta)*(x-1) + 1
        x = torch.relu(x)
        return x
    
    def func_exp_16(self, x):
        x = torch.exp(-torch.exp(2*self.beta)*(torch.abs(x-1)**1.6))
        return x
        
    def func_exp_abs(self, x):
        x = torch.exp(-torch.exp(2*self.beta)*torch.abs(x-1))
        return x
    
    def forward(self, x):
        return self.func_mode(x)
    
class ScaleShift(nn.Module):
    
    def __init__(self, input_dim, scaler_const=False, shifter_const=False, scaler_init=1, shifter_init=0):
        super().__init__()
        self.scaler = scaler_init
        if not scaler_const:
            self.scaler = nn.Parameter(torch.ones(1, input_dim)*scaler_init)
        self.shifter = shifter_init
        if not shifter_const:
            self.shifter = nn.Parameter(torch.ones(1, input_dim)*shifter_init)
        pass
        
    def forward(self, x):
        return x*self.scaler+self.shifter
    

class DistanceTransformBase(nn.Module):
    
    def __init__(self, input_dim, num_centers, p=2):
        super().__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.p = p
        
        self.centers = torch.randn(num_centers, input_dim)/3.
        self.centers = nn.Parameter(self.centers)
    
    def forward(self, x):
        dists = torch.cdist(x, self.centers, p=self.p)
        return dists
    
    def set_centroid_to_data_randomly(self, data_loader):
        indices = np.random.permutation(len(data_loader.dataset.data))[:self.centers.shape[0]]
        self.centers.data = data_loader.dataset.data[indices].to(self.centers.device).reshape(-1, self.centers.shape[1])
        pass
    
    def set_centroid_to_data(self, data_loader):
        new_center = self.centers.data.clone()
        min_dists = torch.ones(self.centers.shape[0])*1e9

        for xx, _ in data_loader:
            xx = xx.reshape(-1, new_center.shape[1])

            dists = torch.cdist(xx, self.centers.data)
            ### min dist of each center to the data points
            min_d, arg_md = dists.min(dim=0)

            ### dont allow same point to be assigned as closest to multiple centroid
            occupied = []
            for i in np.random.permutation(len(arg_md)):
                ind = arg_md[i]
                if ind in occupied:
                    min_d[i] = min_dists[i]
                    arg_md[i] = -1
                else:
                    occupied.append(ind)

            ### the index of centroids that have new min_dist
            idx = torch.nonzero(min_d<min_dists).reshape(-1)

            ### assign new_center to the nearest data point
            new_center[idx] = xx[arg_md[idx]]
            min_dists[idx] = min_d[idx]
            
        self.centers.data = new_center.to(self.centers.device)
        pass

### shift normalized dists towards 0 for sparse activation with exponential
class DistanceTransform_Exp(DistanceTransformBase):
    
    def __init__(self, input_dim, num_centers, p=2, bias=False, eps=1e-5):
        super().__init__(input_dim, num_centers, p=2)
        
        self.scaler = nn.Parameter(torch.ones(1, num_centers)*3/3)
        self.bias = nn.Parameter(torch.ones(1, num_centers)*0) if bias else None
        self.eps = eps
        
    def forward(self, x):
        dists = super().forward(x)
        
        ### normalize similar to UMAP
        dists = dists-dists.mean(dim=1, keepdim=True)
        dists = dists/torch.sqrt(dists.var(dim=1, keepdim=True)+self.eps)
        dists = torch.exp((-dists-2)*self.scaler)
        if self.bias is not None: dists = dists+self.bias
        return dists


### shift normalized dists towards 0 for sparse activation with exponential
class DistanceTransform_MinExp(DistanceTransformBase):
    
    def __init__(self, input_dim, num_centers, p=2, bias=False, eps=1e-5):
        super().__init__(input_dim, num_centers, p=2)
        
        self.scaler = nn.Parameter(torch.ones(1, num_centers)*6/3)
        self.scaler.requires_grad = False
        self.bias = nn.Parameter(torch.ones(1, num_centers)*0) if bias else None
        self.eps = eps
        
    def forward(self, x):
        dists = super().forward(x)
        
        ### normalize similar to UMAP
        dists = dists-dists.min(dim=1, keepdim=True)[0]
        dists = dists/torch.sqrt(dists.var(dim=1, keepdim=True)+self.eps)

        dists = torch.exp(-dists*self.scaler)
        if self.bias is not None: dists = dists+self.bias
        return dists
    
## https://github.com/pswkiki/SphereGAN/blob/master/model_sphere_gan.py
def istereographic(x):
    sqnorm = (x**2).sum(dim=1, keepdim=True) ## l2 norm squared
    x = x*2/(sqnorm+1)
    new_dim = (sqnorm-1)/(sqnorm+1)
    x = torch.cat((x, new_dim), dim=1)
    return x

def stereographic(x):
    x_ = x[:, :-1]
    new_dim = x[:, -1:]
    sqnorm = (1+new_dim)/(1-new_dim)
    x_ = x_/2*(sqnorm+1)
    return x_

class iStereographicLinearTransform(nn.Module):
    
    def __init__(self, input_dim, output_dim, bias=True, normalize=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize
        self.inp_scaler = nn.Parameter(torch.Tensor([np.log(1/np.sqrt(self.input_dim))]))
        self.linear = nn.Linear(input_dim+1, output_dim, bias=bias)
        
        self.linear.weight.data = self.linear.weight.data/self.linear.weight.data.norm(p=2, dim=1, keepdim=True)
        pass
    
    def forward(self, x):
        if self.normalize:
            with torch.no_grad():
                self.linear.weight.data = self.linear.weight.data / self.linear.weight.data.norm(p=2, dim=1, keepdim=True)
                ### linear has weight -> (outdim, indim) format, so normalizing per output dimension
        
        x = x*torch.exp(self.inp_scaler)
        x = self.istereographic(x)
        x = self.linear(x)
        return x
    
    def istereographic(self, x):
        sqnorm = (x**2).sum(dim=1, keepdim=True) ## l2 norm squared
        x = x*2/(sqnorm+1)
        new_dim = (sqnorm-1)/(sqnorm+1)
        x = torch.cat((x, new_dim), dim=1)
        return x

## bias to basic dist
class DistanceTransform(nn.Module):
    
    def __init__(self, input_dim, num_centers, p=2, bias=True):
        super().__init__()
        bias=False
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.p = p
        self.bias = nn.Parameter(torch.zeros(1, num_centers)) if bias else None
        
        self.centers = torch.rand(num_centers, input_dim)
        self.centers = nn.Parameter(self.centers)
        
    def forward(self, x):
#         x = x[:, :self.input_dim]
        dists = torch.cdist(x, self.centers, p=self.p)
        
        ### normalize similar to UMAP
#         dists = dists-dists.min(dim=1, keepdim=True)[0]
        dists = dists-dists.mean(dim=1, keepdim=True)
        dists = dists/dists.std(dim=1, keepdim=True)

        if self.bias is not None: dists = dists+self.bias
        return -dists

    
class EMA(nn.Module):

    def __init__(self, momentum=0.9, mu=None):
        super().__init__()
        if mu is not None:
            mu = torch.Tensor([mu])
        self.register_buffer("mu", mu)
        self.register_buffer("momentum", torch.Tensor([momentum]))

    def __call__(self, x):
        if self.mu is None:
            self.mu = x
        self.mu = self.momentum*self.mu + (1.0 - self.momentum)*x
        return self.mu
    
class DistanceTransformEMA(nn.Module):
    
    def __init__(self, input_dim, num_centers, p=2, bias=True):
        super().__init__()
        bias=False
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.p = p
        self.bias = nn.Parameter(torch.zeros(1, num_centers)) if bias else None
        
        self.centers = torch.rand(num_centers, input_dim)
        self.centers = nn.Parameter(self.centers)
        
        self.std = EMA()
        self.mean = EMA()
        
    def forward(self, x):
        dists = torch.cdist(x, self.centers, p=self.p)
        
        ### normalize similar to UMAP
        mean = self.mean(dists.data.mean(dim=1, keepdim=True).data)
        std = self.std(torch.sqrt(dists.data.var(dim=1, keepdim=True)+1e-5))
        dists = (dists-mean)/std

        if self.bias is not None: dists = dists+self.bias
        return dists
    
class DistanceTransform_MinEMA(nn.Module):
    
    def __init__(self, input_dim, num_centers, p=2, bias=True):
        super().__init__()
        bias=False
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.p = p
        self.bias = nn.Parameter(torch.zeros(1, num_centers)) if bias else None
        
        self.centers = torch.rand(num_centers, input_dim)
        self.centers = nn.Parameter(self.centers)
        
        self.std = EMA()
        self.min = EMA()
        
    def forward(self, x):
#         x = x[:, :self.input_dim]
        dists = torch.cdist(x, self.centers, p=self.p)
        
        ### normalize similar to UMAP
        mean = self.min(dists.data.min(dim=1, keepdim=True)[0].data)
        std = self.std(torch.sqrt(dists.data.var(dim=1, keepdim=True)+1e-5))
        dists = (dists-mean)/std

        if self.bias is not None: dists = dists+self.bias
        return 1-dists


class DistanceTransform_Simple(DistanceTransformBase):
    
    def __init__(self, input_dim, num_centers, p=2, bias=False, eps=1e-5):
        super().__init__(input_dim, num_centers, p=p)
        
        self.scaler = nn.Parameter(torch.ones(1)*0)
        self.bias = nn.Parameter(torch.ones(1, num_centers)*0) if bias else None
        self.eps = eps
        
    def forward(self, x):
        dists = super().forward(x)
        
        dists = (1-dists)*torch.exp(self.scaler)
        
        if self.bias is not None: dists = dists+self.bias
        return dists