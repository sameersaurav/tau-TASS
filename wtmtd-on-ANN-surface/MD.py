import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class FreeEnergyNet(nn.Module):
    def __init__(self):
        super(FreeEnergyNet, self).__init__()
        self.fc1 = nn.Linear(2, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)

    def af(self, x):
        return 1/(1+x**2)
    
    def forward(self, x):
        x = self.af(self.fc1(x))  
        x = self.af(self.fc2(x))
        x = self.fc3(x)
        return x
class double_well_nn:
    
    def __init__(self, model_path):
        """
        """
        self.param = {}
        
        self.model = FreeEnergyNet()  
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  
        
    def act(self, x, y):
        """
        x, y: array-like.
        """
        xy = torch.tensor([[x, y]], dtype=torch.float32)
        
        with torch.no_grad():
            potential = self.model(xy).item()
        
        return potential

    def deriv(self, x, y):
        """
        """
        xy = torch.tensor([[x, y]], dtype=torch.float32, requires_grad=True)
        
        potential = self.model(xy)
        potential.backward()  
        
        dVdx = xy.grad[0, 0].item()  
        dVdy = xy.grad[0, 1].item()
        
        return dVdx, dVdy

class simple_metad:
    def __init__(self, seed=0):
        self._rng = np.random.default_rng(seed)
        self.external_pe = []

    def set_param(self, dt, kT=0.596, damping=1.):
        self.dt = dt
        self.kT = kT
        self.damping = damping

    def set_init_x(self, init_x, init_y, init_vx=0.0, init_vy=0.0):
        self.init_x = init_x
        self.init_y = init_y
        self.init_vx = init_vx
        self.init_vy = init_vy
        list_initial = [0, self.init_x, self.init_y, self.init_vx, self.init_vy]
        self.traj = [list_initial]
        self.vbias = []

    def set_metad(self, init_h, sigma, stride, bias_factor):
        self.init_h = init_h
        self.sigma = sigma
        self.stride = stride
        self.bias_factor = bias_factor
        self.dT = (self.bias_factor - 1) * self.kT

        self.h_t = [self.init_h]
        self.x_t = [self.init_x]
        self.y_t = [self.init_y]

    def dvdt_f(self, x, y, vx, vy):
        grad_x, grad_y = 0, 0
        for pe in self.external_pe:
            gx, gy = pe.deriv(x, y)
            grad_x += gx
            grad_y += gy

        return -grad_x - self.damping * vx, -grad_y - self.damping * vy

    def xi(self, kT, damping):
        """
        Wiener noise.
        """
        sigma = np.sqrt(2 * kT * damping)
        return np.random.normal(loc=0.0, scale=sigma, size=2)

    def periodic_distance(self, d):
        """
        Adjust distances for periodic boundary conditions between -pi and pi.
        """
        return d - 2 * np.pi * np.round(d / (2 * np.pi))
        return adjusted
    
    def flatten_array(arr):
        """Flatten the array and extract scalar values."""
        if isinstance(arr, np.ndarray):
            if arr.ndim == 1:
                return arr.tolist()
            else:
                return arr.flatten().tolist()
        elif isinstance(arr, list):
            flat_list = []
            for item in arr:
                flat_list.extend(flatten_array(item))
            return flat_list
        else:
            return [arr]
    

    def gauss_bias(self, x, y, h_t, x_t, y_t, sigma=0.2):
        x_t = np.array([item[0] if isinstance(item, np.ndarray) else item for item in x_t], dtype=float)
        y_t = np.array([item[0] if isinstance(item, np.ndarray) else item for item in y_t], dtype=float)
        h_t = np.array(h_t, dtype=float)
        
        if len(x_t) != len(y_t):
            raise ValueError("Lengths of x_t and y_t must be the same.")
        
        dx = self.periodic_distance(x - x_t)
        dy = self.periodic_distance(y - y_t)
        
        bias_factor = 5
        h_t = h_t * (bias_factor/(bias_factor-1) )
        gauss_e = h_t * np.exp(-0.5 * (dx**2 + dy**2) / sigma**2)
        gauss_f_x = gauss_e * dx / sigma**2
        gauss_f_y = gauss_e * dy / sigma**2
        return np.sum(gauss_e), np.sum(gauss_f_x), np.sum(gauss_f_y)

    def run(self, nsteps):
        with open('COLVAR', 'w') as f_colvar, open('HILLS', 'w') as f_hills:

            f_colvar.write('time x y metad_bias avg_exp_vbias\n')
            f_hills.write('time x y sigma sigma height bias_factor\n')
    
            bias_exp_sum = 0.0
            cumulative_count = 0
            for i in range(int(nsteps)):
    
                xi = self.xi(self.kT, self.damping)

                i0, x, y, vx, vy = self.traj[-1][0], self.traj[-1][1], self.traj[-1][2], self.traj[-1][3], self.traj[-1][4]
    
                bias_e, bias_f_x, bias_f_y = self.gauss_bias(x, y, self.h_t, self.x_t, self.y_t, self.sigma)

                predict_x = x + self.dt * vx
                predict_y = y + self.dt * vy

                dvdt_f_value_1 = self.dvdt_f(x, y, vx, vy)[0]

                predict_vx = vx + self.dt * self.dvdt_f(x, y, vx, vy)[0] + self.dt * bias_f_x + xi[0] * np.sqrt(self.dt)
                predict_vy = vy + self.dt * self.dvdt_f(x, y, vx, vy)[1] + self.dt * bias_f_y + xi[1] * np.sqrt(self.dt)

                bias_e_pred, bias_f_x_pred, bias_f_y_pred = self.gauss_bias(predict_x, predict_y, self.h_t, self.x_t, self.y_t, self.sigma)

                time = i * self.dt
                
                if i % self.stride == 0:
                    self.x_t.append(x)
                    self.y_t.append(y)
                    if len(self.vbias) > 0:
                        h = self.init_h * np.exp(-self.vbias[-1] / self.dT) * (self.bias_factor/(self.bias_factor-1) )
                    else:
                        h = self.init_h * (self.bias_factor/(self.bias_factor-1) ) 

                    self.h_t.append(h)

                    f_hills.write(f"{time} {x} {y} {self.sigma} {self.sigma} {self.h_t[-1]} {self.bias_factor}\n")
                xf = x + 0.5 * self.dt * (predict_vx + vx)
                yf = y + 0.5 * self.dt * (predict_vy + vy)
                vf_x = vx + 0.5 * self.dt * (self.dvdt_f(predict_x, predict_y, predict_vx, predict_vy)[0] + self.dvdt_f(x, y, vx, vy)[0]) + 0.5 * self.dt * (bias_f_x_pred + bias_f_x) + xi[0] * np.sqrt(self.dt)
                vf_y = vy + 0.5 * self.dt * (self.dvdt_f(predict_x, predict_y, predict_vx, predict_vy)[1] + self.dvdt_f(x, y, vx, vy)[1]) + 0.5 * self.dt * (bias_f_y_pred + bias_f_y) + xi[1] * np.sqrt(self.dt)

                xf = (xf + np.pi) % (2 * np.pi) - np.pi
                yf = (yf + np.pi) % (2 * np.pi) - np.pi

                list_xf_yf_vf_x_vf_y = [i, xf, yf, vf_x, vf_y]
                self.traj.append(list_xf_yf_vf_x_vf_y)
                self.vbias.append(0.5 * (bias_e + bias_e_pred))

                bias_exp_sum += np.exp((self.vbias[-1])/ self.kT)
                cumulative_count += 1
                avg_exp_vbias = bias_exp_sum / cumulative_count
                if i % 100 == 0 :
                    f_colvar.write("{} {} {} {} {} \n".format(time, x, y, self.vbias[-1], avg_exp_vbias))

                if 0.5 <= xf <= 1.5:
                    print(f'Stopping criteria met')
                    break

def metad_model(sigma):
    """
    Generate a metadynamics simulation with Gaussian bias width = sigma.
    """
    nn_model_path = 'free_energy.pt'  
    dw = double_well_nn(model_path=nn_model_path)  
    metad = simple_metad(seed=5)
    metad.set_param(dt=0.002, kT=0.596, damping=1.)
    metad.set_init_x(-1.44, 1.16)
    metad.set_metad(init_h=0.3, sigma=0.25, stride=10000, bias_factor=5.)
    metad.external_pe = [dw]
    metad.run(1e9)

sigmas = [0.25]
for idx, sigma in enumerate(sigmas):
    metad_model(sigma)

