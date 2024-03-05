import lmdb
import torch.utils.data as data
import torchvision

class BingdingDataset(data.Dataset):
    def __init__(self, 
                 lmdb_path,
                 transform:torchvision.transforms.transforms.Compose=None,):
        """
        Create a dataset from the LMDB file,
        the LMDB dataset must have the following keys:
        'atoms': the name of the atoms in the molecule
        'coordinates': the coordinates of the atoms in the molecule
        'pocket_atoms': the name of the atoms in the pocket
        'pocket_coordinates': the coordinates of the atoms in the pocket
        """
        self.data = self._load_lmdb(lmdb_path)

    def _load_lmdb(self, lmdb_path):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

