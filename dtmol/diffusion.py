import numpy as np
import torch
from dtmol.utils.so3_op import sample, sample_vec, score_vec, score_norm
from copy import copy
from scipy.spatial.transform import Rotation
from typing import Callable, Union, List
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
        return x.cpu().detach().numpy()
    except:
        return x
    
def try_to_tensor(x:Union[torch.Tensor,np.ndarray]):
    if isinstance(x,torch.Tensor):
        return x.to(torch.float32)
    else:
        return torch.tensor(x,dtype = torch.float32)

def is_int(x):
    return isinstance(x,int) or isinstance(x,np.integer)

def is_scalar(x):
    return is_int(x) or x.ndim == 0    

def expand_as(x,y):
    B,N,D = y.shape
    if is_scalar(x):
        x = x*np.ones((B,N,D),dtype = int)
    if x.ndim == 1:
        x = x[:,None,None]*np.ones((B,N,D),dtype = int)
    return x

def s_normal(*shape):
    return np.random.normal(0,1,shape)

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

    
    @property
    def sigma(self):
        #used in VP-SDE format
        return np.sqrt(1 - self.alpha)

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

class LogLinearScheduler(NoiseSchedular):
    def __init__(self,
                 T:int,
                 sigma_min:float = 1e-5,
                 sigma_max:float = 0.999):
        super().__init__(T, sigma_min, sigma_max)
    
    def __call__(self,t):
        return self.sigma_min * (self.sigma_max/self.sigma_min)**self._t(t)

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
        return (1 - self._t(t)**2) * self.sigma_max

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
                 sde_format = "VP",
                 return_negative_score = False,
                 schedular:Callable = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.T = T
        self.sde_format = sde_format
        self.conjugate_sampler = None
        self.return_negative_score = return_negative_score
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
    
    def set_T(self, new_T):
        # Set a new maximum time T for the sampler.
        self.T = new_T
        self.schedular.T = new_T
        self.load_schedular(self.schedular)

    def sample_given_t_ve(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        raise NotImplementedError
    
    def sample_given_t_vp(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        raise NotImplementedError

    def sample_given_t(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        if self.sde_format == "VP":
            return self.sample_given_t_vp(x,t)
        elif self.sde_format == "VE":
            return self.sample_given_t_ve(x,t)
        else:
            raise ValueError("The SDE type should be either 'VP' or 'VE'")

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
            if self.sde_format == "VP":
                assert np.all(self.alphas>=0), "Use VP-SDE format, but found <0 alpha, please check the max sigma is setting < 1."
            self.sigma = schedular.sigma
        else:
            self.noise = np.asarray([schedular(t) for t in range(self.T)])
            self.alphas = alpha_series(self.noise)
            self.sigma = np.sqrt(1 - self.alphas)

    def sample(self, x):
        raise NotImplementedError

    def __call__(self,x,ts = None):
        #Also accept chaining call for the sampler.
        if ts:
            return self.sample_given_t(x,ts)
        else:
            return self.sample(x)
    
    @staticmethod
    def _vp_kernel(x_t, score, beta_t, alpha_t, with_noise = True, form = "pc"):
        if form == 'direct':
            x_rev = (np.maximum(np.sqrt(1-beta_t),3))(beta_t*score + x_t)
            #direct form if we reverse the discrete diffusion, note this form would diverge at beta_t = 1
        elif form == 'pc':
            x_rev = (2 - np.sqrt(1 - beta_t))*x_t + beta_t * score
            #form given by Algorithm 3 in Song et. al. https://arxiv.org/pdf/2011.13456.pdf
        elif form == 'tylor2':
            x_rev = (1+0.5*beta_t + 0.75 * beta_t**2)*x_t + (beta_t + 0.5 * beta_t**2) * score
            #form by expand to the second order of direct form
        elif form == 'ddpm':
            x_rev = (1/np.sqrt(1-beta_t))*(x_t+beta_t/(np.sqrt(1-alpha_t))*score)
        if with_noise:
            x_rev = x_rev + np.sqrt(beta_t) * s_normal(*x_t.shape)
        return x_rev

    
    @staticmethod
    def _ve_kernel(x_t, score, sigma_t, sigma_t_1, with_noise = True):
        var = (sigma_t**2 - sigma_t_1**2)
        if score.ndim == 3 and var.ndim == 2:
            var = var[...,None]
        x_rev = x_t + var * score
        if with_noise:
            x_rev = x_rev + np.sqrt(var) * s_normal(*x_t.shape)
        return x_rev

    def reverse_dt(self,
                   x:torch.tensor,
                   t:int,
                   score:torch.tensor,
                   stochastic:bool = False):
        x = try_to_numpy(x)
        score = try_to_numpy(score)
        if self.sde_format == "VP":
            return self.reverse_vp_dt(x = x,
                                      t = t,
                                      score = score,
                                      stochastic=stochastic)
        elif self.sde_format == "VE":
            return self.reverse_ve_dt(x = x,
                                      t = t,
                                      score = score,
                                      stochastic=stochastic)
        else:
            raise ValueError("The SDE type should be either 'VP' or 'VE'")

class DummySampler(BaseSampler):
    def __init__(self,
                 T:int = 5000,
                 seed = None,
                 sde_format = "VP",
                 return_negative_score = False,
                 system_wise = False,
                 schedular:Callable = None):
        """This is a dummy sampler that add no noise to the input coordinates.
        """
        super().__init__(T, seed, sde_format, return_negative_score, schedular)
        self.system_wise = system_wise
        np.random.seed(seed)

    def sample_given_t(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        B,N,D = x.shape
        if self.system_wise:
            score = np.zeros((B,D))
            norm = np.ones((B,1))
        else:
            score = np.zeros((B,N,D))
            norm = np.ones((B,N))
        return x, score, norm
    
    def sample(self, x:torch.tensor):
        x = try_to_tensor(x)
        if x.dim() != 3:
            raise ValueError("Expecting input tensor to have shape (B,N,D), but got shape {}".format(x.shape))
        B = x.shape[0]
        ts = self.sample_time(B)
        return *self.sample_given_t(x,ts),ts

    def reverse_dt(self,
                   x:torch.tensor,
                   t:int,
                   score:torch.tensor,
                   stochastic:bool = False):
        return x

class GaussianSampler(BaseSampler):
    def __init__(self,
                 T:int = 5000,
                 seed = None,
                 time_sync:bool = True,
                 sde_format = "VE",
                 return_negative_score = False,
                 schedular:Callable = None):
        super().__init__(T, seed, sde_format, return_negative_score, schedular)
        self.time_sync = time_sync
        np.random.seed(seed)

    def kernel(self,
               t:Union[int,np.ndarray],
               x_t:np.ndarray,
               x_0:np.ndarray):
        """Return the log probability of the Gaussian kernel: log p_t(x_t | x_0).
        """
        if is_int(t):
            t = [t] * len(x_t)
        alpha_t = self.alphas[t]
        mu = np.sqrt(alpha_t)*x_0
        variance = 1 - alpha_t
        return -0.5 * np.log(2 * np.pi) - np.log(variance) - 0.5 * (x_t - mu) ** 2 / variance

    def score(self,eps,sampled):
        return eps

    def sample_given_t_vp(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        """
        Input:
            x: Union[torch.tensor,np.ndarray], the input coordinates of the atoms, shape (N,3).
            t: Union[int,torch.Tensor,np.ndarray], the time step, can be set differently for each atom.
        """
        x = try_to_numpy(x)
        B,N,D = x.shape
        x_c = x.mean(axis = 1,keepdims = True)
        x = x - x_c
        if is_int(t):
            t = t*np.ones((B,N),dtype = int)
        if t.ndim == 1:
            t = t[:,None]*np.ones((B,N),dtype = int)
        e = s_normal(B,N,D)
        variance = 1 - self.alphas[t]
        variance = variance[...,None]
        scale = np.sqrt(self.alphas[t])[...,None]
        x_t =  scale * x + np.sqrt(variance) * e + x_c
        score = -self.score(e,x_t) if self.return_negative_score else self.score(e,x_t)
        norm = np.squeeze(np.sqrt(variance),axis = -1)
        return torch.tensor(x_t), score, norm

    def sample_given_t_ve(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        """
        Input:
            x: Union[torch.tensor,np.ndarray], the input coordinates of the atoms, shape (N,3).
            t: Union[int,torch.Tensor,np.ndarray], the time step, can be set differently for each atom.
        """
        x = try_to_numpy(x)
        B,N,D = x.shape
        if is_int(t):
            t = t*np.ones((B,N),dtype = int)
        if t.ndim == 1:
            t = t[:,None]*np.ones((B,N),dtype = int)
        e = s_normal(B,N,D)
        variance = self.noise[t]**2
        variance = variance[...,None]
        x_t =  x + np.sqrt(variance) * e
        score = -self.score(e,x_t) if self.return_negative_score else self.score(e,x_t)
        norm = np.squeeze(np.sqrt(variance),axis = -1)
        return torch.tensor(x_t), score, norm

    def sample_given_t(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        if self.sde_format == "VP":
            return self.sample_given_t_vp(x,t)
        elif self.sde_format == "VE":
            return self.sample_given_t_ve(x,t)
        else:
            raise ValueError("The SDE type should be either 'VP' or 'VE'")

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
    
    def reverse_vp_dt(self, 
                x:torch.tensor, 
                t:int, 
                score:torch.tensor,
                stochastic:bool = False):
        x = try_to_numpy(x)
        B,N,D = x.shape
        if is_int(t):
            t = t*np.ones((B,N),dtype = int)
        if t.ndim == 1:
            t = t[:,None]*np.ones((B,N),dtype = int)
        beta_t = self.noise[t][...,None]
        alpha_t = self.alphas[t][...,None]
        if stochastic:
            e = s_normal(B,N,D)
        else:
            e = 0
        score = score if self.return_negative_score else -score
        # apply reverse perturbation to the coordinates
        x_rev = self._vp_kernel(x,score,beta_t,alpha_t,with_noise = stochastic)
        return x_rev

    def reverse_ve_dt(self,
                      x:torch.tensor,
                      t:int,
                      score:torch.tensor,
                      stochastic:bool = False):
        x = try_to_numpy(x)
        B,N,D = x.shape
        if is_int(t):
            t = t*np.ones((B,N),dtype = int)
        if t.ndim == 1:
            t = t[:,None]*np.ones((B,N),dtype = int)
        sigma_t = self.noise[t][...,None]
        sigma_t_1 = self.noise[np.maximum(t-1,0)][...,None]
        sigma_t_1[t==0] = 0
        score = score if self.return_negative_score else -score
        x_rev = self._ve_kernel(x,score,sigma_t,sigma_t_1,with_noise = stochastic)
        return x_rev

class RotationSampler(BaseSampler):
    def __init__(self,
                 T:int = 5000,
                 seed = None,
                 sde_format = "VE",
                 return_negative_score = True,
                 schedular:Callable = None):
        assert return_negative_score, "Rotation sampler should return negative score."
        super().__init__(T, seed, sde_format, return_negative_score, schedular)
        np.random.seed(seed)

    def kernel(self,
               t:Union[int,np.ndarray],
               x_t:np.ndarray):
        """Return the log probability of the Gaussian kernel: log p_t(x_t | x_0).
        """
        raise NotImplementedError

    def score(self,eps,sampled,normalize = True):
        if normalize:
            return score_vec(eps,sampled)/score_norm(eps)
        else:
            return score_vec(eps,sampled)
    
    def sample_given_t_vp(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        """
        Input:
            x: Union[torch.tensor,np.ndarray], the input coordinates of the atoms, shape (N,3).
        """
        x = try_to_tensor(x)
        t = try_to_numpy(t)
        b,N,D = x.shape
        assert D==3, "Rotation sampler works only for 3D coordinates"
        if is_int(t):
            t = np.asarray([t] * b)
        variance = 1 - self.alphas[t]
        eps = np.sqrt(variance)
        eular_vec = np.vstack([sample_vec(eps[i])for i in range(b)])
        score = np.vstack([self.score(e,vec) for e,vec in zip(eps,eular_vec)])
        score = score if self.return_negative_score else -score #score is already negative score.
        norm = score_norm(eps)[...,None]
        with torch.no_grad():
            Rot = torch.Tensor(Rotation.from_rotvec(eular_vec).as_matrix())
            x_c = x.mean(axis = 1,keepdims = True)
            x_t = torch.einsum('ijk,ilk->ilj',Rot,(x - x_c))+ x_c
        return x_t, score, norm

    def sample_given_t_ve(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        """
        Input:
            x: Union[torch.tensor,np.ndarray], the input coordinates of the atoms, shape (N,3).
            t: Union[int,torch.Tensor,np.ndarray], the diffusion time for the batch.
        """
        x = try_to_tensor(x)
        t = try_to_numpy(t)
        b,N,D = x.shape
        assert D==3, "Rotation sampler works only for 3D coordinates"
        if is_int(t):
            t = np.asarray([t] * b)
        variance = self.noise[t]**2
        eps = np.sqrt(variance)
        eular_vec = np.vstack([sample_vec(eps[i])for i in range(b)])
        score = np.vstack([self.score(e,vec) for e,vec in zip(eps,eular_vec)])
        score = score if self.return_negative_score else -score #score is already negative score.
        norm = score_norm(eps)[...,None]
        with torch.no_grad():
            Rot = torch.Tensor(Rotation.from_rotvec(eular_vec).as_matrix())
            x_c = x.mean(axis = 1,keepdims = True)
            x_t = torch.einsum('ijk,ilk->ilj',Rot,(x - x_c))+ x_c
        return x_t, score, norm

    def dimensional_check(self,x,score):
        B,N,D = x.shape
        score_shape = score.shape
        assert len(score_shape) == 2, "The score should be a system score with two dimensions (B,3)"
        B1,D1 = score_shape
        assert B == B1, "The batch size of the input coordinates and the score should be the same."
        assert D1 == D, "The score should have the same dimension as the input coordinates."

    def sample(self, x:Union[torch.tensor,np.ndarray]):
        x = try_to_tensor(x)
        if x.dim() != 3:
            raise ValueError(f"Expecting input tensor to have shape (B,N,D), but got shape {x.shape}") 
        b = x.shape[0]
        ts = self.sample_time(size = b)
        return *self.sample_given_t(x,ts),ts
    
    def reverse_ve_dt(self, 
                x:torch.tensor, 
                t:int, 
                score:torch.tensor,
                stochastic:bool = False):
        #only VE-SDE form can be used for rotation matrix, ref: Algorithm 4  https://arxiv.org/pdf/2210.01776.pdf
        #notice the score being predicted is already the negative one.
        x = try_to_tensor(x)
        t = try_to_numpy(t)
        B,N,D = x.shape
        self.dimensional_check(x,score)
        assert D==3, "Rotation sampler works only for 3D coordinates"
        if is_int(t):
            t = np.asarray([t] * B)
        assert t.ndim == 1, "The time step should be a 1D array with shaep (B)"
        sigma_t = self.noise[t][...,None] # (B,1)
        sigma_t_1 = self.noise[np.maximum(t-1,0)][...,None] # (B,1)
        sigma_t_1[t==0] = 0
        score = score if self.return_negative_score else -score
        r_t = self._ve_kernel(np.zeros((B,D)),score,sigma_t,sigma_t_1,with_noise = stochastic)
        with torch.no_grad():
            Rot = torch.Tensor(Rotation.from_rotvec(r_t).as_matrix())
            x_c = torch.nanmean(x,axis = 1,keepdims = True)
            x_rev = torch.einsum('ijk,ilk->ilj',Rot,(x - x_c)) + x_c
        return x_rev
    
    def reverse_vp_dt(self, 
                x:torch.tensor, 
                t:int, 
                score:torch.tensor,
                stochastic:bool = False):
        raise NotImplementedError('VP-SDE form is not applicable for rotation matrix')

class TranslationSampler(BaseSampler):
    def __init__(self,
                 T:int = 5000,
                 seed = None,
                 sde_format = "VP",
                 return_negative_score = False,
                 schedular:Callable = None):
        """The translation sampler. Which would diffuse the mean of the input coordinates.
        from x_c:t_0 -> norm(0,I):t_T
        """
        super().__init__(T, seed,sde_format, return_negative_score, schedular)
        np.random.seed(seed)

    def kernel(self,
               t:Union[int,np.ndarray],
               x_t:np.ndarray,
               x_0:np.ndarray):
        """Return the log probability of the Gaussian kernel: log p_t(x_t | x_0).
        """
        if is_int(t):
            t = [t] * len(x_t)
        alpha_t = self.alphas[t]
        mu = np.sqrt(alpha_t)*x_0
        variance = 1 - alpha_t
        return -0.5 * np.log(2 * np.pi) - np.log(variance) - 0.5 * (x_t - mu) ** 2 / variance

    def score(self,eps,sampled):
        return eps

    def sample_given_t_vp(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        """
        Input:
            x: Union[torch.tensor,np.ndarray], the input coordinates of the atoms, shape (B,N,3).
            t: Union[int,torch.Tensor,np.ndarray], the time step, can be set differently for each atom.
        """
        x = try_to_numpy(x)
        B,N,D = x.shape
        if is_int(t):
            t = [t] * B
        e = s_normal(len(x),D)
        variance = 1 - self.alphas[t]
        variance = variance[...,None]
        scale = np.sqrt(self.alphas[t])[...,None]
        with torch.no_grad():
            x_c = x.mean(axis = 1)
            x_c_diff =  scale * x_c + np.sqrt(variance) * e
            x_t = x - x_c + x_c_diff
            score = self.score(e,x_t)
            score = -score if self.return_negative_score else score
        return torch.tensor(x_t), score, np.sqrt(variance)

    def sample_given_t_ve(self,x:Union[torch.tensor,np.ndarray],t:Union[int,torch.Tensor,np.ndarray]):
        """
        Input:
            x: Union[torch.tensor,np.ndarray], the input coordinates of the atoms, shape (B,N,3).
            t: Union[int,torch.Tensor,np.ndarray], the time step, can be set differently for each atom.
        """
        x = try_to_numpy(x)
        B,N,D = x.shape
        if is_int(t):
            t = [t] * B
        e = s_normal(len(x),D)
        std = self.noise[t]
        std = std[...,None]
        with torch.no_grad():
            x_t = x + std * e
            score = self.score(e,x_t)
            score = -score if self.return_negative_score else score
        #norm would be used in loss by 1/norm, but instead we want score*norm, 
        #so we return reverse of the norm (1/std), result in l = (score-pred)**2*std**2
        #which make the loss bigger when the std is large, in order to make model to 
        #focuse more on the high noise region.
        return torch.tensor(x_t), score, 1/std

    def sample(self, x:torch.tensor):
        x = try_to_tensor(x)
        if x.dim() != 3:
            raise ValueError("Expecting input tensor to have shape (B,N,D), but got shape {}".format(x.shape))
        b = x.shape[0]
        ts = self.sample_time(size = b)
        return *self.sample_given_t(x,ts),ts
    
    def dimensional_check(self,x,score):
        B,N,D = x.shape
        score_shape = score.shape
        assert len(score_shape) == 2, "The score should be a system score with two dimensions (B,3), but got shape {}".format(score_shape)
        B1,D1 = score_shape
        assert B == B1, "The batch size of the input coordinates and the score should be the same."
        assert D1 == D, "The score should have the same dimension as the input coordinates."

    def reverse_vp_dt(self, 
                x:torch.tensor, 
                t:int, 
                score:torch.tensor,
                stochastic:bool = False):
        x = try_to_numpy(x)
        B,N,D = x.shape
        self.dimensional_check(x,score)
        assert len(score.shape) == 2, "The score should be a system score with two dimensions (B,3)"
        if is_int(t):
            t = np.asarray([t] * B)
        beta_t = self.noise[t][...,None] # (B,1)
        alpha_t = self.alphas[t][...,None] # (B,1)
        x_c = np.nanmean(x,axis = 1) # (B,D)
        score = score if self.return_negative_score else -score # (B,D)
        x_c_rev = self._vp_kernel(x_c,score,beta_t,alpha_t,with_noise = stochastic)
        x_t = x - x_c[:,None,:] + x_c_rev[:,None,:]
        return torch.tensor(x_t)
    
    def reverse_ve_dt(self,
                      x:torch.tensor,
                      t:int,
                      score:torch.tensor,
                      stochastic:bool = False):
        x = try_to_numpy(x)
        B,N,D = x.shape
        self.dimensional_check(x,score)
        if is_int(t):
            t = np.asarray([t] * B)
        sigma_t = self.noise[t][:,None]
        sigma_t_1 = self.noise[np.maximum(t-1,0)][:,None]
        sigma_t_1[t==0] = 0
        #Algorithm 3 predictor part in PC sampling (VE SDE) in Song et al., https://arxiv.org/pdf/2011.13456.pdf
        score = score if self.return_negative_score else -score
        x_c = np.nanmean(x,axis = 1)
        x_c_diff = self._ve_kernel(np.zeros_like(x_c),score,sigma_t,sigma_t_1,with_noise = stochastic)
        x_t = x + x_c_diff[:,None,:]
        return torch.tensor(x_t)    

class ChainSampler(BaseSampler):
    def __init__(self, sampler:BaseSampler):
        self.samplers = [sampler]
    
    @property
    def T(self):
        return self.samplers[0].T

    @property
    def rng(self):
        return self.samplers[0].rng

    def set_T(self, new_T):
        for sampler in self.samplers:
            sampler.set_T(new_T)

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
            score_temp = score_temp[:,None,:] if score_temp.ndim == 2 else score_temp
            norm_temp = norm_temp[:,None] if norm_temp.ndim == 1 else norm_temp
            score = np.concatenate([score,score_temp],axis = 1)
            norm = np.concatenate([norm,norm_temp],axis = 1)
        return x_t,score,norm,ts
    
    def reverse_dt(self,
                   x:torch.tensor,
                   t:int,
                   scores:List[torch.tensor],
                   stochastic = False):
        x_t = x
        for sampler,score in zip(self.samplers[::-1],scores[::-1]):
            x_t = sampler.reverse_dt(x = x_t,
                                     t = t,
                                     score = score,
                                     stochastic = stochastic)
        return x_t

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
    T = 500
    test_nan = True
    linear_sch = LinearScheduler(T)
    ll_sch_std = LogLinearScheduler(T,sigma_min = 1e-5, sigma_max = 1) #Parameter value from diffdock rot_sigma_min/max
    ll_sch_tr = LogLinearScheduler(T,sigma_min = 0.1, sigma_max = 3) #Parameter value ~ box radius
    ll_sch_rot = LogLinearScheduler(T,sigma_min = 0.1, sigma_max = 1.65)
    ll_sch_pert = LogLinearScheduler(T,sigma_min = 0.1, sigma_max = 2)
    cos_sch = CosineScheduler(T)
    geo_sch = GeometricScheduler(T)
    poly_sch = PolynomialScheduler(T)
    fig,ax = plt.subplots(1,2,figsize = (10,5))
    ax[0].plot(linear_sch.noise,label = "Linear")
    ax[0].plot(cos_sch.noise,label = "Cosine")
    ax[0].plot(geo_sch.noise,label = "Geometric")
    ax[0].plot(poly_sch.noise,label = "Polynomial")
    ax[0].plot(ll_sch_std.noise,label = "LogLinear std")
    ax[0].set_title("Noise")
    ax[0].set_ylabel("beta_t")
    ax[0].legend()
    ax[1].plot(linear_sch.alpha,label = "Linear")
    ax[1].plot(cos_sch.alpha,label = "Cosine")
    ax[1].plot(geo_sch.alpha,label = "Geometric")
    ax[1].plot(poly_sch.alpha,label = "Polynomial")
    ax[1].plot(ll_sch_std.alpha,label = "LogLinear std")
    ax[1].set_title("Alpha")
    ax[1].legend()
    ax[1].set_ylabel("alpha_t")
    
    rot_sampler = RotationSampler(T = T, schedular=ll_sch_rot)
    g_sampler = GaussianSampler(T = T, schedular = ll_sch_pert)
    g_sampler2 = GaussianSampler(T = T,schedular = ll_sch_pert)
    tr_sampler = TranslationSampler(T = T,schedular = ll_sch_tr,sde_format = "VE")
    # tr_sampler = TranslationSampler(T = T,schedular = ll_sch_std,sde_format = "VP")
    composed = ChainSampler(rot_sampler).compose(tr_sampler).compose(g_sampler)
    composed1 = ChainSampler(g_sampler2)
    #generate a mesh grid
    x = np.linspace(-1,1,5)
    y = np.linspace(-1,1,5)
    z = np.linspace(-1,1,5)
    x,y,z = np.meshgrid(x,y,z)
    x_0 = np.vstack([x.flatten(),y.flatten(),z.flatten()]).T
    x_0 = torch.tensor(x_0).unsqueeze(0)

    # Diffuse the input coordinates
    x_1, score, norm,ts = rot_sampler.sample(x_0)
    x_2, g_score, g_norm,g_ts = g_sampler.sample(x_1)
    x_3, tr_score, tr_norm,tr_ts = tr_sampler.sample(x_2)
    composed1.conjugate(composed)
    x_compose, c_score, c_norm,c_ts = composed.sample(x_0)

    actual_disp = x_compose[0].mean(dim = 0) - x_0[0].mean(dim = 0)
    score_disp = c_score[:,1]
    print("Actual displacement:",actual_disp)
    print("Predicted displacement:",score_disp)
    correlation = np.corrcoef(actual_disp,score_disp)
    print("Correlation:",correlation)

    # Reverse the diffusion
    reverse_T = 20
    rot_sampler.set_T(reverse_T)
    g_sampler.set_T(reverse_T)
    tr_sampler.set_T(reverse_T)
    composed.set_T(reverse_T)
    dist_compose, dist_rot, dist_g, dist_tr = [],[],[],[]
    reverse_with_stochastic = False
    def get_reverse_ts(t,T = 20, old_T = 5000):
        t = t * T // old_T
        return np.arange(t,-1,-1)

    def average_distances(x,x_):
        return np.mean(np.linalg.norm(x - x_,axis = -1))

    ts_rev = get_reverse_ts(c_ts,T = reverse_T,old_T = T)
    if test_nan:
        x_compose[0,0,:] = np.nan
    x_rev = x_compose
    for t in ts_rev:
        dist_compose.append(average_distances(x_0,x_rev))
        x_rev = composed.reverse_dt(x_rev,
                                    t,
                                    [c_score[:,0,:],c_score[:,1,:],c_score[:,2:,:]],
                                    stochastic = reverse_with_stochastic)
        #here rotation score is already negative, so we need to negative it back.
    dist_compose.append(average_distances(x_0,x_rev))

    ts_rev = get_reverse_ts(ts,T = reverse_T,old_T = T)
    x_0rev = x_1
    for t in ts_rev:
        dist_rot.append(average_distances(x_0,x_0rev))
        x_0rev = rot_sampler.reverse_dt(x_0rev,int(t),score,stochastic=reverse_with_stochastic)
    dist_rot.append(average_distances(x_0,x_0rev))
    
    ts_rev = get_reverse_ts(g_ts,T = reverse_T,old_T = T)
    x_1rev = x_2
    for t in ts_rev:
        dist_g.append(average_distances(x_1,x_1rev))
        x_1rev = g_sampler.reverse_dt(x_1rev,int(t),g_score,stochastic=reverse_with_stochastic)
        dist_g.append(average_distances(x_1,x_1rev))
    dist_g.append(average_distances(x_1,x_1rev))

    ts_rev = get_reverse_ts(tr_ts,T = reverse_T,old_T = T)
    x_2rev = x_3
    for t in ts_rev:
        dist_tr.append(average_distances(x_2,x_2rev))
        x_2rev = tr_sampler.reverse_dt(x_2rev,int(t),tr_score,stochastic=reverse_with_stochastic)
    dist_tr.append(average_distances(x_2,x_2rev))

    _,_,_,c1_ts = composed1.sample(x_0)
    x_0,x_1,x_2,x_3,x_c = x_0[0],x_1[0],x_2[0],x_3[0],x_compose[0]
    x_rev, x_0rev,x_1rev,x_2rev = x_rev[0],x_0rev[0],x_1rev[0],x_2rev[0]
    
    # Create a figure for all subplots
    fig = plt.figure(figsize=(20, 20))
    # First subplot: Sampling
    ax1 = fig.add_subplot(221, projection='3d')  # 2x2 grid, position 1
    ax1.scatter(x_0[:,0], x_0[:,1], x_0[:,2], color='r', label="x_0")
    ax1.scatter(x_c[:,0], x_c[:,1], x_c[:,2], color='c', label="x_compose")
    ax1.scatter(x_rev[:,0], x_rev[:,1], x_rev[:,2], color='m', label="x_rev")
    ax1.legend()
    # Second subplot: Rotation sampler
    ax2 = fig.add_subplot(222, projection='3d')  # 2x2 grid, position 2
    ax2.scatter(x_0[:,0], x_0[:,1], x_0[:,2], color='r', label="x_0")
    ax2.scatter(x_1[:,0], x_1[:,1], x_1[:,2], color='b', label="x_rot")
    ax2.scatter(x_0rev[:,0], x_0rev[:,1], x_0rev[:,2], color='m', label="x_rev")
    ax2.legend()
    # Third subplot: Gaussian sampler
    ax3 = fig.add_subplot(223, projection='3d')  # 2x2 grid, position 3
    ax3.scatter(x_1[:,0], x_1[:,1], x_1[:,2], color='b', label="x_rot")
    ax3.scatter(x_2[:,0], x_2[:,1], x_2[:,2], color='g', label="x_g")
    ax3.scatter(x_1rev[:,0], x_1rev[:,1], x_1rev[:,2], color='m', label="x_rev")
    ax3.legend()
    # Fourth subplot: Translation sampler
    ax4 = fig.add_subplot(224, projection='3d')  # 2x2 grid, position 4
    ax4.scatter(x_2[:,0], x_2[:,1], x_2[:,2], color='g', label="x_2")
    ax4.scatter(x_3[:,0], x_3[:,1], x_3[:,2], color='y', label="x_tr")
    ax4.scatter(x_2rev[:,0], x_2rev[:,1], x_2rev[:,2], color='m', label="x_rev")
    ax4.legend()
    # Display the figure
    plt.show()

    # Create a figure for all subplots
    fig = plt.figure(figsize=(20, 20))
    # plot the distance of the reverse diffusion
    ax1 = fig.add_subplot(221)  # 2x2 grid, position 1
    ax1.plot(dist_rot,label = "Rotation")
    plt.legend()
    
    ax2 = fig.add_subplot(222)  # 2x2 grid, position 2
    ax2.plot(dist_g,label = "Gaussian")
    plt.legend()

    ax3 = fig.add_subplot(223)  # 2x2 grid, position 3
    ax3.plot(dist_tr,label = "Translation")
    plt.legend()

    ax4 = fig.add_subplot(224)  # 2x2 grid, position 4
    ax4.plot(dist_compose,label = "Composed")
    plt.legend()