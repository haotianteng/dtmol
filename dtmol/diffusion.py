import numpy as np
import torch
from dtmol.utils.so3_op import sample, sample_vec, score_vec, score_norm
from copy import copy
from scipy.spatial.transform import Rotation
from typing import Callable, Union
from functools import lru_cache

def alpha_series(noise: np.ndarray):
    """
    Calculating the `alpha_t = \prod_{t=1}^{T} (1 - \eta_t)` series.
    Input:
    - noise: np.ndarray, the noise series.
    """
    alpha = np.cumprod(1 - noise)
    return alpha

def try_to_numpy(x:Union[torch.Tensor,np.ndarray]):
    try: 
        return x.numpy()
    except:
        return x
    
def try_to_tensor(x:Union[torch.Tensor,np.ndarray]):
    if isinstance(x,torch.Tensor):
        return x.to(torch.float32)
    else:
        return torch.tensor(x,dtype = torch.float32)

class NoiseSchedular(object):
    def __init__(self,
                 T:int,
                 sigma_min:float,
                 sigma_max:float,
                 eps = 1e-5):
        self.T = T
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = eps
    
    def __call__(self,t):
        raise NotImplementedError

    @property
    def noise(self): #beta_t
        return np.asarray([self(t) for t in range(self.T)])

    @property
    def alpha(self):
        return alpha_series(self.noise)
    
    def _t(self,t):
        assert t >= 0 and t < self.T, f"t should be in the range [0,{self.T-1}], but got {t}"
        return (t+1)/(self.T)

class LinearScheduler(NoiseSchedular):
    def __init__(self, 
                 T:int, 
                 sigma_min:float = 1e-5,
                 sigma_max:float = 0.999):
        super().__init__(T, sigma_min, sigma_max)

    def __call__(self,t):
        return self.sigma_min + (self.sigma_max - self.sigma_min) * self._t(t)
    
class CosineScheduler(NoiseSchedular):
    """Cosine noise scheduler from https://arxiv.org/pdf/2102.09672.pdf
    """
    def __init__(self, 
                 T:int):
        super().__init__(T, sigma_min = 1e-3, sigma_max = 0.999, eps = 8e-3)

    def __call__(self,t):
        return min(self.noise[t], self.sigma_max)

    def _alpha_bar(self,t):
        return self._f(t)/self._f(0)
    
    def _f(self,t):
        return np.cos((t/self.T+self.eps)/(1+self.eps)*np.pi/2)**2

    @property
    @lru_cache(maxsize=1)
    def _alpha(self):
        return np.asarray([self._alpha_bar(t) for t in range(self.T+1)])
    
    @property
    def alpha(self):
        return self._alpha[1:]
    
    @property
    @lru_cache(maxsize=1)
    def noise(self):
        #beta_t = 1 - alpha_t/alpha_{t-1}
        return 1 - self._alpha[1:]/self._alpha[:-1]

class PolynomialScheduler(NoiseSchedular):
    """Cosine noise scheduler from https://arxiv.org/pdf/2102.09672.pdf
    """
    def __init__(self, 
                 T:int,
                 eps = 1e-5):
        super().__init__(T, 1e-3, 0.999)
        self.eps = 1e-5

    def __call__(self,t):
        return min(self.noise[t], self.sigma_max)

    def _alpha_bar(self,t):
        return 1 - self._t(t)**2

    @property
    def alpha(self):
        return np.asarray([self._alpha_bar(t) for t in range(self.T)])
    
    @property
    @lru_cache(maxsize=1)
    def noise(self):
        #beta_t = 1 - alpha_t/alpha_{t-1}
        return np.concatenate([[1-self.alpha[0]],1 - self.alpha[1:]/self.alpha[:-1]])

class GeometricScheduler(NoiseSchedular):
    """The schedular for VE-SDE style diffusion model proposed in 
    Song et al., https://arxiv.org/pdf/2011.13456.pdf
    """
    def __init__(self,
                 T:int,
                 sigma_min:float = 1e-5,
                 sigma_max:float = 0.999):
        super().__init__(T, sigma_min, sigma_max)
    
    def __call__(self,t):
        #This is faster than the original implementation
        return self.sigma_min * (self.sigma_max/self.sigma_min)**self._t(t)
    
    # def __call__(self,t):
    #     return self.sigma_min**(1-t/self.T) * self.sigma_max**(t/self.T)

class BaseSampler(object):
    """Base class for samplers.
    Input parameters:
    - T: int, the number of time steps.
    - seed: int, the random seed.
    - sceduler: Callable, the sceduler function, default will use a linear sceduler.
    = conjugation: BaseSampler, the conjugate sampler.
    Usage:
        sampler.sample(x) -> x_t, score, norm, ts
        sampler.sample_given_t(x,t) -> x_t, score, norm
            where x is the input coordinates of the atoms, shape (B,N,D).
            will return the sampled x_t, the score and the norm of the diffusion.
            x_t: the sampled coordinates of the atoms, ahs the same shape as input x_0 (B,N,D).
            score: the score of the diffusion, shape (B,N,D) if sampled noise for every atom or (B,1,D) for system noise.
            norm: the norm of the diffusion, shape (B,N) or (B,1) according to the shape of the score.
    """
    def __init__(self,
                 T:int = 5000,
                 seed:int = None,
                 schedular:Callable = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.T = T
        self.conjugate_sampler = None
        if schedular is None:
            schedular = lambda t: 1/(self.T-1)
        self.load_schedular(schedular)

    def kernel(self,
               t:int,
               x_t:np.ndarray,
               x_0:np.ndarray):
        raise NotImplementedError

    def sample_time(self, size:Union[int,tuple]):
        # Comment out because this only works for Python >= 3.0 and numpy >= 1.25
        # if self.conjugate_sampler:
        #     if self.rng.bit_generate.state != self.conjugate_sampler.rng.bit_generate.state:
        #         print("Warning, sampler is no longer synchronized with conjugation.")
        return self.rng.choice(self.T,size = size)
        
    def conjugate(self,sampler):
        """Synchronize the time with the given sampler.
        """
        self.rng = copy(sampler.rng)
        self.conjugate_sampler = sampler

    def score(self,eps,sampled):
        raise NotImplementedError

    def load_schedular(self, schedular: Callable):
        self.schedular = schedular
        if isinstance(schedular,NoiseSchedular):
            self.schedular.T = self.T
            self.noise = schedular.noise
            self.alphas = schedular.alpha
        else:
            self.noise = np.asarray([schedular(t) for t in range(self.T)])
            self.alphas = alpha_series(self.noise)

    def sample(self, x):
        raise NotImplementedError
    
    def sample_given_t(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        raise NotImplementedError

    def __call__(self,x,ts = None):
        #Also accept chaining call for the sampler.
        if ts:
            return self.sample_given_t(x,ts)
        else:
            return self.sample(x)


class GaussianSampler(BaseSampler):
    def __init__(self,
                 T:int = 5000,
                 seed = None,
                 time_sync:bool = True,
                 schedular:Callable = None):
        super().__init__(T, seed, schedular)
        self.time_sync = time_sync
        np.random.seed(seed)

    def kernel(self,
               t:Union[int,np.ndarray],
               x_t:np.ndarray,
               x_0:np.ndarray):
        """Return the log probability of the Gaussian kernel: log p_t(x_t | x_0).
        """
        if isinstance(t,int):
            t = [t] * len(x_t)
        alpha_t = self.alphas[t]
        mu = np.sqrt(alpha_t)*x_0
        variance = 1 - alpha_t
        return -0.5 * np.log(2 * np.pi) - np.log(variance) - 0.5 * (x_t - mu) ** 2 / variance

    def score(self,eps,sampled):
        return eps

    def sample_given_t(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        """
        Input:
            x: Union[torch.tensor,np.ndarray], the input coordinates of the atoms, shape (N,3).
            t: Union[int,torch.Tensor,np.ndarray], the time step, can be set differently for each atom.
        """
        x = try_to_numpy(x)
        B,N,D = x.shape
        x_c = x.mean(axis = 1)
        x = x - x_c[:,None,:]
        if isinstance(t,int):
            t = t*np.ones((B,N),dtype = int)
        if t.ndim == 1:
            t = t[:,None]*np.ones((B,N),dtype = int)
        e = np.random.normal(0,1,(B,N,D))
        variance = 1 - self.alphas[t]
        variance = variance[...,None]
        scale = np.sqrt(self.alphas[t])[...,None]
        x_t =  scale * x + np.sqrt(variance) * e + x_c[:,None,:]
        score = self.score(e,x_t)
        norm = np.squeeze(np.sqrt(variance),axis = -1)
        return torch.tensor(x_t), score, norm
    
    def sample(self, x:torch.tensor):
        x = try_to_tensor(x)
        if x.dim() != 3:
            raise ValueError(f"Expecting input tensor to have shape (B,N,D), but got shape {x.shape}") 
        B,N,D = x.shape
        if self.time_sync:
            ts = self.sample_time(B)
        else:
            ts = self.sample_time((B,N))
        return *self.sample_given_t(x,ts),ts

class RotationSampler(BaseSampler):
    def __init__(self,
                 T:int = 5000,
                 seed = None,
                 schedular:Callable = None):
        super().__init__(T, seed, schedular)
        np.random.seed(seed)

    def kernel(self,
               t:Union[int,np.ndarray],
               x_t:np.ndarray):
        """Return the log probability of the Gaussian kernel: log p_t(x_t | x_0).
        """
        raise NotImplementedError

    def score(self,eps,sampled):
        return score_vec(eps,sampled)
    
    def sample_given_t(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        """
        Input:
            x: Union[torch.tensor,np.ndarray], the input coordinates of the atoms, shape (N,3).
        """
        x = try_to_tensor(x)
        t = try_to_numpy(t)
        b,N,D = x.shape
        assert D==3, "Rotation sampler works only for 3D coordinates"
        if isinstance(t,int):
            t = np.asarray([t] * b)
        variance = 1 - self.alphas[t]
        eps = np.sqrt(variance)
        eular_vec = np.vstack([sample_vec(eps[i])for i in range(b)])
        score = np.vstack([self.score(e,vec) for e,vec in zip(eps,eular_vec)])
        norm = score_norm(eps)[...,None]
        with torch.no_grad():
            Rot = torch.Tensor(Rotation.from_rotvec(eular_vec).as_matrix())
            x_c = x.mean(axis = 1)
            x_t = torch.einsum('ijk,ilk->ilj',Rot,(x - x_c))+ x_c
        return x_t, score, norm
    
    def sample(self, x:Union[torch.tensor,np.ndarray]):
        x = try_to_tensor(x)
        if x.dim() != 3:
            raise ValueError(f"Expecting input tensor to have shape (B,N,D), but got shape {x.shape}") 
        b = x.shape[0]
        ts = self.sample_time(size = b)
        return *self.sample_given_t(x,ts),ts

class TranslationSampler(BaseSampler):
    def __init__(self,
                 T:int = 5000,
                 seed = None,
                 schedular:Callable = None):
        """The translation sampler. Which would diffuse the mean of the input coordinates.
        from x_c:t_0 -> norm(0,I):t_T
        """
        super().__init__(T, seed, schedular)
        np.random.seed(seed)

    def kernel(self,
               t:Union[int,np.ndarray],
               x_t:np.ndarray,
               x_0:np.ndarray):
        """Return the log probability of the Gaussian kernel: log p_t(x_t | x_0).
        """
        if isinstance(t,int):
            t = [t] * len(x_t)
        alpha_t = self.alphas[t]
        mu = np.sqrt(alpha_t)*x_0
        variance = 1 - alpha_t
        return -0.5 * np.log(2 * np.pi) - np.log(variance) - 0.5 * (x_t - mu) ** 2 / variance

    def score(self,eps,sampled):
        return eps

    def sample_given_t(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        """
        Input:
            x: Union[torch.tensor,np.ndarray], the input coordinates of the atoms, shape (B,N,3).
            t: Union[int,torch.Tensor,np.ndarray], the time step, can be set differently for each atom.
        """
        x = try_to_numpy(x)
        B,N,D = x.shape
        if isinstance(t,int):
            t = [t] * B
        e = np.random.randn(len(x),D)
        variance = 1 - self.alphas[t]
        variance = variance[:,None]
        scale = np.sqrt(self.alphas[t])[:,None]
        with torch.no_grad():
            x_c = x.mean(axis = 1)
            x_c_diff =  scale * x_c + np.sqrt(variance) * e
            x_t = x - x_c[:,None,:] + x_c_diff[:,None,:]
            score = self.score(e,x_t)
        return torch.tensor(x_t), score[:,None,:], np.sqrt(variance)
    
    def sample(self, x:torch.tensor):
        x = try_to_tensor(x)
        if x.dim() != 3:
            raise ValueError("Expecting input tensor to have shape (B,N,D), but got shape {}".format(x.shape))
        b = x.shape[0]
        ts = self.sample_time(size = b)
        return *self.sample_given_t(x,ts),ts

class ChainSampler(BaseSampler):
    def __init__(self, sampler:BaseSampler):
        self.samplers = [sampler]
    
    @property
    def T(self):
        return self.samplers[0].T

    @property
    def rng(self):
        return self.samplers[0].rng

    def compose(self, sampler:BaseSampler):
        assert self.T == sampler.T, "The max time T of the samplers should be the same."
        sampler.rng = self.rng
        self.samplers.append(sampler)
        return self

    def sample(self,x):
        x_t,score,norm,ts = self.samplers[0].sample(x)
        score = score[:,None,:] if score.ndim == 2 else score
        norm = norm[:,None] if norm.ndim == 1 else norm
        for sampler in self.samplers[1:]:
            x_t,score_temp,norm_temp = sampler.sample_given_t(x_t,ts) 
            score = np.concatenate([score,score_temp],axis = 1)
            norm = np.concatenate([norm,norm_temp],axis = 1)
        return x_t,score,norm,ts
    
    def conjugate(self,sampler):
        for s in self.samplers:
            s.conjugate(sampler)

    def __repr__(self):
        string = "ChainSampler("
        for sampler in self.samplers:
            string += f"{sampler.__class__.__name__} -> "
        string = string[:-4] + ")"
        return string
    
if __name__ == "__main__":
    #Test scheduler
    from matplotlib import pyplot as plt
    T = 5000
    linear_sch = LinearScheduler(T)
    cos_sch = CosineScheduler(T)
    geo_sch = GeometricScheduler(T)
    poly_sch = PolynomialScheduler(T)
    fig,ax = plt.subplots(1,2,figsize = (10,5))
    ax[0].plot(linear_sch.noise,label = "Linear")
    ax[0].plot(cos_sch.noise,label = "Cosine")
    ax[0].plot(geo_sch.noise,label = "Geometric")
    ax[0].plot(poly_sch.noise,label = "Polynomial")
    ax[0].set_title("Noise")
    ax[0].set_ylabel("beta_t")
    ax[0].legend()
    ax[1].plot(linear_sch.alpha,label = "Linear")
    ax[1].plot(cos_sch.alpha,label = "Cosine")
    ax[1].plot(geo_sch.alpha,label = "Geometric")
    ax[1].plot(poly_sch.alpha,label = "Polynomial")
    ax[1].set_title("Alpha")
    ax[1].legend()
    ax[1].set_ylabel("alpha_t")
    
    rot_sampler = RotationSampler(schedular=geo_sch)
    g_sampler = GaussianSampler(schedular = poly_sch)
    g_sampler2 = GaussianSampler(schedular = poly_sch)
    tr_sampler = TranslationSampler(schedular = cos_sch)
    composed = ChainSampler(rot_sampler).compose(g_sampler).compose(tr_sampler)
    composed1 = ChainSampler(g_sampler2)
    #generate a mesh grid
    x = np.linspace(-1,1,10)
    y = np.linspace(-1,1,10)
    z = np.linspace(-1,1,10)
    x,y,z = np.meshgrid(x,y,z)
    x_0 = np.vstack([x.flatten(),y.flatten(),z.flatten()]).T
    x_0 = torch.tensor(x_0).unsqueeze(0)
    x_1, score, norm,ts = rot_sampler.sample(x_0)
    x_2, g_score, g_norm,g_ts = g_sampler.sample(x_1)
    x_3, tr_score, tr_norm,tr_ts = tr_sampler.sample(x_2)
    composed1.conjugate(composed)
    x_compose, c_score, c_norm,c_ts = composed.sample(x_0)
    _,_,_,c1_ts = composed1.sample(x_0)
    print(c_ts,c1_ts)
    # 3D plot the sampled x
    fig = plt.figure(figsize = (10,10))
    x_0,x_1,x_2,x_3,x_c = x_0[0],x_1[0],x_2[0],x_3[0],x_compose[0]
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_0[:,0], x_0[:,1], x_0[:,2], color='r',label = "x_0", )
    # ax.scatter(x_1[:,0], x_1[:,1], x_1[:,2], color='b',label = "x_rot")
    ax.scatter(x_2[:,0], x_2[:,1], x_2[:,2], color='g',label = "x_g")
    ax.scatter(x_3[:,0], x_3[:,1], x_3[:,2], color='y',label = "x_tr")
    # ax.scatter(x_c[:,0], x_c[:,1], x_c[:,2], color='c',label = "x_compose")
    ax.legend()
    plt.show()
    composed