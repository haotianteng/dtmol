## this script including utilities that is used to do reverse process of the diffusion process (sampling process)
import torch
import numpy as np
from dtmol.utils.padding import get_padding_mask, get_padding_mask2d

def update_distance_matrix(coor,mole_padding,prot_padding):
    """
    Update the distance matrix given the new coordinates of the atoms
    coor: torch.tensor, shape (batch, n_mole+n_prot, 3) Coordinates of the molecules and proteins
    mole_padding: torch.tensor, shape (batch, n_mole) The padding mask for the molecules
    prot_padding: torch.tensor, shape (batch, n_prot) The padding mask for the molecules
    """ 
    dist = torch.cdist(coor,coor,p=2)
    padding_mask = get_padding_mask2d(mole_padding,prot_padding)
    dist[padding_mask] = 0
    return dist

def _reverse_sampling(coord, scores, rev_sampler, padding, t, stochastic=False):
    """
    Reverse the sampling process of the diffusion process
    coord: torch.tensor, shape (batch, n_atoms, 3) The coordinates of the atoms
    scores: List[torch.tensor], shape (batch, n_atoms, 3) The score matrix of the atoms
    rev_sampler: dtmol.diffusion.ReverseSampler The reverse sampler
    padding: torch.tensor, shape (batch, n_mole+n_prot) The padding mask for the atoms
    t: int The time step of the reverse sampling
    """
    coor = rev_sampler.reverse_dt(coord, t, scores, stochastic=stochastic)
    if not torch.is_tensor(coor):
        coor = torch.tensor(coor,device=coord.device,dtype=coord.dtype)
    else:
        coor = coor.to(coord.device)
    return coor

def reverse_sampling(coord, score, mole_sampler, prot_sampler, mole_padding, prot_padding, t, stochastic=False):
    n_mole = mole_padding.size(1)
    mole_score = [score[:,0],score[:,1],score[:,2:n_mole+2]] # Translation score, rotation score, and the perturbation score
    prot_score = [score[:,n_mole+2:]] # Protein only has the perturbation score
    coor_mole = _reverse_sampling(coord[:,:n_mole], mole_score, mole_sampler, mole_padding, t, stochastic=stochastic)
    coor_prot = _reverse_sampling(coord[:,n_mole:], prot_score, prot_sampler, prot_padding, t, stochastic=stochastic)
    coor_padding = get_padding_mask(mole_padding,prot_padding)
    coor = torch.cat([coor_mole,coor_prot],dim=1)
    coor[coor_padding] = torch.nan
    dist = update_distance_matrix(coor, mole_padding=mole_padding, prot_padding=prot_padding)
    return coor, dist
        
def rmsd(coord, label, mole_padding, prot_padding):
    """ Calculate the RMSD between the predicted coordinates and the true coordinates
    coord: torch.tensor, shape (batch, n_mole+n_prot, 3) The predicted coordinates
    label: torch.tensor, shape (batch, n_mole+n_prot, 3) The true coordinates
    mole_padding: torch.tensor, shape (batch, n_mole) The padding mask for the molecules
    prot_padding: torch.tensor, shape (batch, n_prot) The padding mask for the molecules
    """
    diff = coord - label
    padding_mask = get_padding_mask(mole_padding,prot_padding)
    n_mole = mole_padding.size(1)
    mole_padding = padding_mask[:,:n_mole]
    prot_padding = padding_mask[:,n_mole:]
    diff[padding_mask,:] = 0
    diff = diff**2 # (batch_n, n_atoms, 3)
    diff = torch.sqrt(diff.sum(dim=-1)) # (batch_n, n_atoms)
    diff_mole = diff[:,:n_mole].sum(dim=-1)/(~mole_padding).sum(dim=-1) # (batch_n)
    diff_prot = diff[:,n_mole:].sum(dim=-1)/(~prot_padding).sum(dim=-1) # (batch_n)
    return diff_mole, diff_prot

if __name__ == "__main__":
    #testing code
    pass