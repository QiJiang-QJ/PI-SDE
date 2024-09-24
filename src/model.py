import torch
from collections import OrderedDict
from torch import nn
import src.sde as sde






class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

class MLP(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(input_dim, hidden_dim), LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(hidden_dim, hidden_dim))
            model.append(LipSwish())
        model.append(torch.nn.Linear(hidden_dim, out_dim))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)

class AutoGenerator(nn.Module):

    def __init__(self, config):
        super(AutoGenerator, self).__init__()

        self.dim = config.x_dim
        self.k_dims = config.k_dims
        self.layers = config.layers
        self.sigma_type = config.sigma_type
        self.sigma_const = config.sigma_const

        self.activation = config.activation
        if self.activation == 'relu':
            self.act = nn.LeakyReLU
        elif self.activation == 'softplus':
            self.act = nn.Softplus
        elif self.activation == 'tanh':
            self.act = nn.Tanh
        elif self.activation == 'none':
            self.act = None
        else:
            raise NotImplementedError
        
        self.net_ = []
        for i in range(self.layers): 
            if i == 0: 
                self.net_.append(('linear{}'.format(i+1), nn.Linear(self.dim+1, self.k_dims[i]))) 
            else: 
                self.net_.append(('linear{}'.format(i+1), nn.Linear(self.k_dims[i-1], self.k_dims[i]))) 
            if self.activation == 'none': 
                pass
            else:
                self.net_.append(('{}{}'.format(self.activation, i+1), self.act()))
        self.net_.append(('linear', nn.Linear(self.k_dims[-1], 1, bias = False)))
        self.net_ = OrderedDict(self.net_)
        self.net = nn.Sequential(self.net_)

        net_params = list(self.net.parameters())
        net_params[-1].data = torch.zeros(net_params[-1].data.shape) 

        self.noise_type = 'diagonal'
        self.sde_type = "ito"

        cfg = dict(
            input_dim=self.dim + 1,
            out_dim=self.dim,
            hidden_dim=128,
            num_layers=2,
            tanh=True
        )
        if self.sigma_type == 'Mlp':
            self.sigma = MLP(**cfg)
        elif self.sigma_type == "const":
            self.register_buffer('sigma', torch.as_tensor(self.sigma_const))
            self.sigma = self.sigma.repeat(self.dim).unsqueeze(0)
        elif self.sigma_type == "const_param":
            self.sigma = nn.Parameter(torch.randn(1,self.dim), requires_grad=True)

    def _pot(self, xt):
        xt = xt.requires_grad_()
        pot = self.net(xt)
        return pot

    def f(self, t, x_r):
        x = x_r[:,0:-1] 
        t = (((torch.ones(x.shape[0]).to(x.device)) * t).unsqueeze(1))
        xt = torch.cat([x, t], dim=1)
        pot = self._pot(xt)                    # batch * 1
        drift = torch.autograd.grad(pot, xt, torch.ones_like(pot),create_graph=True)[0]

        drift_x = -drift[:,0:-1]               # batch * N
        drift_t = drift[:,-1].unsqueeze(1)     # batch *1

        delta_hjb = torch.abs(drift_t - 0.5 * torch.sum(torch.pow(drift_x, 2),1,keepdims=True))
        new_drift = torch.cat([drift_x, delta_hjb], dim=1)  # batch * (N+1)
        return new_drift
    
    def _drift(self, xt):
        pot = self._pot(xt)                    # batch * 1
        drift = torch.autograd.grad(pot, xt, torch.ones_like(pot),create_graph=True)[0]

        drift_x = -drift[:,0:-1]                # batch * N
        return drift_x

    def g(self, t, x_r):
        x = x_r[:,0:-1]
        if self.sigma_type == "Mlp": 
            t = (((torch.ones(x.shape[0]).to(x.device)) * t).unsqueeze(1))
            xt = torch.cat([x, t], dim=1)
            g = self.sigma(xt).view(-1, self.dim)
        elif self.sigma_type == "const":
            g = self.sigma.repeat(x.shape[0], 1)
        elif self.sigma_type == "const_param":
            g = self.sigma.repeat(x.shape[0], 1)
        extra_g = (((torch.zeros(x.shape[0]).to(x.device))).unsqueeze(1))
        g = torch.cat([g, extra_g], dim=1)
        return g





class ForwardSDE(torch.nn.Module):
    def __init__(self, config):
        super(ForwardSDE, self).__init__()

        self._func = AutoGenerator(config)

    def forward(self, ts, x_r_0):
        x_r_s = sde.sdeint_adjoint(self._func, x_r_0, ts, method=self.solver, dt=0.1, dt_min=0.0001,
                                     adjoint_method='euler', names={'drift': 'f', 'diffusion': 'g'} )
        return x_r_s


