## this script including utilities that is used to do reverse process of the diffusion process (sampling process)
import numpy as np
def get_reverse_ts(t,T = 20, old_T = 5000):
    t = t * T // old_T
    return np.arange(t,-1,-1)

class ReverseSampler:
    def __init__(self, 
                 config,
                 forward_sampler,
                 score_network):
        self.forward_sampler = forward_sampler
        self.score_network = score_network
        self.config = config

    def prepare_input(self,batch):
        protein_input = {batch['net_input']}
    
    def sample(T = 20，batch):
        
