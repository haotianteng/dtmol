import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name+

def get_padding_index(padding, val = True, order = "last"):
    """
    Get the index of the padding token in the padding mask
    padding: torch.tensor, shape (batch, n) The padding mask
    val: bool, whether the padding token is True or False
    order: str, "last" or "first", whether to get the first or last padding toekn of val
    """
    val = torch.tensor(val)
    assert order in ["last","first"], "order must be 'last' or 'first'"
    assert padding.dim() == 2, "padding must be 2d tensor"
    if order == "last":
        padding = padding.flip(dims=[1])
    idxs = (padding == val).int().argmax(dim=1)
    if order == "last":
        idxs = padding.size(1)-1-idxs
    return idxs


def get_padding_mask(mole_padding, prot_padding,pad_special_token = True):
    """
    Get the padding mask for the molecules and proteins
    mole_padding: torch.tensor, shape (batch, n_mole) The padding mask for the molecules
    prot_padding: torch.tensor, shape (batch, n_prot) The padding mask for the molecules
    pad_special_token: bool, whether to pad the special token
    """
    mole_padding, prot_padding = mole_padding.clone(), prot_padding.clone()
    if pad_special_token:    
        # get the location of the <e> token for each sample by get the last False value in the padding mask
        mole_e_token = get_padding_index(mole_padding, val = False, order = "last")
        prot_e_token = get_padding_index(prot_padding, val = False, order = "last")
        mole_padding[torch.arange(mole_padding.size(0)),mole_e_token] = True
        prot_padding[torch.arange(prot_padding.size(0)),prot_e_token] = True
        mole_padding[:,0] = True
        prot_padding[:,0] = True # <s> token
    padding_mask = torch.cat([mole_padding,prot_padding],dim=1) 
    return padding_mask

def get_padding_mask2d(mole_padding, prot_padding, pad_special_token = True):
    """
    Get the 2d padding mask matrix for the molecules and proteins
    mole_padding: torch.tensor, shape (batch, n_mole) The padding mask for the molecules
    prot_padding: torch.tensor, shape (batch, n_prot) The padding mask for the molecules
    pad_special_token: bool, whether to pad the special token
    """
    padding_mask = get_padding_mask(mole_padding, prot_padding,pad_special_token)
    padding_mask = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
    return padding_mask

if __name__ == "__main__":
    test_mole_padding = torch.tensor([[False,False,False,True,True],
                                        [False,False,False,False,False]])
    test_prot_padding = torch.tensor([[False,False,False,True,True],
                                        [False,False,False,False,True]])
    print(get_padding_mask(test_mole_padding,test_prot_padding))
    print(get_padding_mask2d(test_mole_padding,test_prot_padding))