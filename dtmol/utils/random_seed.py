import torch
import contextlib
@contextlib.contextmanager
def torch_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    orig_seed = torch.seed()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.manual_seed(orig_seed)

if __name__ == "__main__":
    with torch_seed(1,2,3):
        print(torch.randint(0,10,(3,3))) #this should be the same
    print(torch.randint(0,10,(3,3))) #this should be different