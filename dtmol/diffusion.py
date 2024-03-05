import numpy as np
import torch
from utils.so3_op import sample, sample_vec, score_vec, score_norm
from typing import Callable, Union

def alpha_series(noise: torch.Tensor):
    """
    Calculating the `alpha_t = \prod_{t=1}^{T} (1 - \eta_t)` series.
    Input:
    - noise: np.ndarray, the noise series.
    """
    alpha = torch.cumprod(1 - noise)
    return alpha

class BaseSampler(object):
    """Base class for samplers.
    Input parameters:
    - T: int, the number of time steps.
    - seed: int, the random seed.
    - sceduler: Callable, the sceduler function, default will use a linear sceduler.
    """
    def __init__(self,
                 T:int = 5000,
                 seed:int = 0,
                 sceduler:Callable = None):
        self.seed = seed
        self.T = T
        if sceduler is not None:
            scedular = lambda t: (t+1)/self.T
        self.load_schedular(scedular)

    def kernel(self,
               t:int,
               x_t:np.ndarray,
               x_0:np.ndarray):
        raise NotImplementedError

    def score(self,eps,sampled):
        raise NotImplementedError

    def load_schedular(self, schedular: Callable):
        self.schedular = schedular
        self.noise = np.asarray([schedular(t) for t in range(self.T)])
        self.alphas = alpha_series(self.noise)

    def sample_once(self, t):
        raise NotImplementedError

class GaussianSampler(BaseSampler):
    def __init__(self,
                 T:int = 5000,
                 seed = None,
                 sceduler:Callable = None):
        super().__init__(T, seed, sceduler)
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

    def sample_given_t(self,x:torch.tensor,t:Union[int,torch.Tensor,np.ndarray]):
        b = len(x)
        if isinstance(t,int):
            t = [t] * b
        e = torch.randn(len(x))
        variance = 1 - self.alphas[t]
        scale = np.sqrt(self.alphas[t])
        x_t =  scale * x + np.sqrt(variance) * e
        score = self.score(e,x_t)
        return x_t, score
    
    def sample(self, x:torch.tensor):
        b = x.shape[0]
        ts = np.random.choice(self.T, size = b)
        return self.sample_given_t(x,ts)

class RotationSampler(BaseSampler):
    def __init__(self,
                 T:int = 5000,
                 seed = None,
                 sceduler:Callable = None):
        super().__init__(T, seed, sceduler)
        np.random.seed(seed)

    def kernel(self,
               t:Union[int,np.ndarray],
               x_t:np.ndarray,
               x_0:np.ndarray):
        """Return the log probability of the Gaussian kernel: log p_t(x_t | x_0).
        """

    def score(self,eps,sampled):
        return score_vec(eps,sampled)
    

if __name__ == "__main__":
    sampler = GaussianSampler()
    x = torch.tensor([1,2,3,4,5])
    x_t, score = sampler.sample(x)
    print(x_t, score)