import lmdb
import numpy as np
import os
import pickle
from functools import lru_cache
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawArrayDataset,
    FromNumpyDataset,
)
from unimol.data import (
    KeyDataset,
    DistanceDataset,
    EdgeTypeDataset,
    NormalizeDataset,
    RightPadDatasetCoord,
    ConformerSampleConfGDataset,
    ConformerSampleConfGV2Dataset,
    data_utils,
)
from unicore.data import (
    EpochShuffleDataset,
) # Additional load for protein dataset
from unimol.data import (
    ConformerSamplePocketDataset,
    MaskPointsPocketDataset,
    CroppingPocketDataset,
    AtomTypeDataset,
) # Additional load for protein dataset

from torch.utils.data import Dataset
from argparse import Namespace
from typing import Dict


class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        return data

class MoleculeDataset(object):
    """Read the molecular dataset"""
    def __init__(self,dictionary,config:Dict):
        self.dictionary = dictionary
        #transfer config to namespace
        self.config = Namespace(**config)
        self.datasets = {}
    def load_lmdb(self,path,split):
        """Load LMDB dataset from path."""
        split_path = os.path.join(path, f"{split}.lmdb")
        dataset = LMDBDataset(split_path)
        smi_dataset = KeyDataset(dataset, "smi")
        src_dataset = KeyDataset(dataset, "atoms")
        if not split.startswith("test"):
            sample_dataset = ConformerSampleConfGV2Dataset(
                dataset,
                self.config.seed,
                "atoms",
                "coordinates",
                "target",
                self.config.beta,
                self.config.smooth,
                self.config.topN,
            )
        else:
            sample_dataset = ConformerSampleConfGDataset(
                dataset, self.config.seed, "atoms", "coordinates", "target"
            )
        sample_dataset = NormalizeDataset(sample_dataset, "coordinates")
        sample_dataset = NormalizeDataset(sample_dataset, "target")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.config.max_seq_len
        )
        coord_dataset = KeyDataset(sample_dataset, "coordinates")
        tgt_coord_dataset = KeyDataset(sample_dataset, "target")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        tgt_coord_dataset = FromNumpyDataset(tgt_coord_dataset)
        tgt_coord_dataset = PrependAndAppend(tgt_coord_dataset, 0.0, 0.0)
        tgt_distance_dataset = DistanceDataset(tgt_coord_dataset)

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "coord_target": RightPadDatasetCoord(
                        tgt_coord_dataset,
                        pad_idx=0,
                    ),
                    "distance_target": RightPadDataset2D(
                        tgt_distance_dataset,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
            },
        )
        if split.startswith("train"):
            with data_utils.numpy_seed(self.config.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset


class ProteinDataset(object):
    """Read the protein pocket dataset, the class was modified from the unimol/tasks/unimol_pocket.py"""
    def __init__(self, dictionary, args):
        self.dictionary = dictionary
        self.args = Namespace(**args)
        self.datasets = {}
        self.dict_type = self.args.dict_type
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)

    def load_lmdb(self, path, split):
        split_path = os.path.join(path, f"{split}.lmdb")
        raw_dataset = LMDBDataset(split_path)
        def one_dataset(raw_dataset, coord_seed, mask_seed):
            pdb_id_dataset = KeyDataset(raw_dataset, "pdbid")
            dataset = ConformerSamplePocketDataset(
                raw_dataset, coord_seed, "atoms", "coordinates", self.dict_type
            )
            dataset = AtomTypeDataset(raw_dataset, dataset)
            dataset = CroppingPocketDataset(
                dataset, self.args.seed, "atoms", "coordinates", self.args.max_atoms
            )
            dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
            token_dataset = KeyDataset(dataset, "atoms")
            token_dataset = TokenizeDataset(
                token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
            )
            coord_dataset = KeyDataset(dataset, "coordinates")
            residue_dataset = KeyDataset(dataset, "residue")
            expand_dataset = MaskPointsPocketDataset(
                token_dataset,
                coord_dataset,
                residue_dataset,
                self.dictionary,
                pad_idx=self.dictionary.pad(),
                mask_idx=self.mask_idx,
                noise_type=self.args.noise_type,
                noise=self.args.noise,
                seed=mask_seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
            )

            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)

            encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
            encoder_target_dataset = KeyDataset(expand_dataset, "targets")
            encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")

            src_dataset = PrependAndAppend(
                encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
            )
            tgt_dataset = PrependAndAppend(
                encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
            )
            encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
            encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)

            edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
            coord_dataset = FromNumpyDataset(coord_dataset)
            coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
            distance_dataset = DistanceDataset(coord_dataset)
            return {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": RightPadDatasetCoord(
                    encoder_coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": RightPadDataset2D(
                    encoder_distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
            }, {
                "tokens_target": RightPadDataset(
                    tgt_dataset, pad_idx=self.dictionary.pad()
                ),
                "distance_target": RightPadDataset2D(distance_dataset, pad_idx=0),
                "coord_target": RightPadDatasetCoord(coord_dataset, pad_idx=0),
                "pdb_id": RawArrayDataset(pdb_id_dataset),
            }

        net_input, target = one_dataset(raw_dataset, self.args.seed, self.args.seed)
        dataset = {"net_input": net_input, "target": target}
        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

if __name__ == "__main__":
    protein_dict = Dictionary.load("/home/haotiant/diffMol/models/pretrain/unimol_protein_dict.txt")
    protein_path = "/data/unimol_data/pockets/"
    test_config = {
        "seed": 0,
        "beta": 0.1,
        "smooth": 0.1,
        "topN": 10,
        "max_seq_len": 100,
        "dict_type": "coarse",
        "max_atoms": 256,
        "noise_type": "normal", #noise type in coordinate noise, can be "trunc_normal", "uniform", "normal", "none"
        "noise": 1.,#coordinate noise for masked atoms
        "mask_prob":0.15, #probability of replacing a token with mask
        "leave_unmasked_prob":0.1, #probability that a masked token is unmasked
        "random_token_prob":0.05, #probability of replacing a mask token with a random token
    }
    pocket_dataset = ProteinDataset(protein_dict,test_config)
    pocket_dataset.load_lmdb(protein_path,"train")