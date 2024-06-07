import lmdb
import numpy as np
import os
import pickle
from dtmol.utils.random_seed import torch_seed
from functools import lru_cache
from unicore.data import (
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

from unimol.data import (
    ConformerSampleDockingPoseDataset,
    CroppingPocketDockingPoseDataset,
    RemoveHydrogenPocketDataset,
    PrependAndAppend2DDataset,
    NormalizeDockingPoseDataset,
    CrossDistanceDataset,
    RightPadDatasetCross2D,
) # Additional load for cross (mole + protein) dataset

from dtmol.utils.dataset_util import (
    DisplacementDataset,
    CrossDisplacementDataset,
    FullDisplacementDataset,
    RightPadDataset3D,
    RightPadDatasetCross3D,
    PrependAndAppend3DDataset,
) # Customized dataset for displacement

from unicore.data import BaseWrapperDataset
import torch
from torch.utils.data import Dataset
from argparse import Namespace
from typing import Dict


class DictDataset(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets = {}

    def __getitem__(self, key):
        return self.datasets.__getitem__(key)

    def __setitem__(self, key, value):
        return self.datasets.__setitem__(key, value)

    def __delitem__(self, key):
        return self.datasets.__delitem__(key)

    def __len__(self):
        return self.datasets.__len__()

    def __contains__(self, key):
        return self.datasets.__contains__(key)

    def keys(self):
        return self.datasets.keys()

    def values(self):
        return self.datasets.values()

    def items(self):
        return self.datasets.items()

    def get(self, key, default=None):
        return self.datasets.get(key, default)

    def clear(self):
        return self.datasets.clear()

    def update(self, other_dict):
        return self.datasets.update(other_dict)

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

class CrossEdgeTypeDataset:
    def __init__(self, mol_dataset, pocket_dataset,num_types:int):
        self.mol_dataset = mol_dataset
        self.pocket_dataset = pocket_dataset
        self.num_types = num_types
    
    def __len__(self):
        return len(self.mol_dataset)

    @lru_cache(maxsize=1)
    def __getitem__(self, idx):
        source = self.mol_dataset[idx].clone()
        target = self.pocket_dataset[idx].clone()
        edge_type = source.view(-1, 1) * self.num_types + target.view(1, -1)
        return edge_type

class DiffusionDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        diffuser,
        is_train = True,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.set_epoch(1)
        self.diffuser = diffuser
        self.seed = seed
        self.istrain = is_train
    
    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=128) #This LRU cache must be set, so that diffusion sampling returns score that is consistent with the diffused coordinates
    def __cached_item__(self, index: int, epoch: int):
        item = np.array(self.dataset[index])[None,...]
        with data_utils.numpy_seed(self.seed, epoch, index), torch_seed(self.seed, epoch, index):
            diffused,score,norm,time_steps = self.diffuser(item)
            self.seed = torch_seed(self.seed, epoch, index)
        return {"diffused": diffused[0].to(torch.float32), "score": score[0],"norm":norm[0], "time_steps":time_steps}
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
    
class SliceDataset(BaseWrapperDataset):
    def __init__(self, dataset, start = None, end = None):
        super().__init__(dataset)
        self.start = start
        self.end = end

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=1)
    def __getitem__(self, index):
        item = self.dataset[index]
        if self.start:
            item = item[self.start:]
        if self.end:
            item = item[:self.end]
        return item

class MoleculeDataset(DictDataset):
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
            src_dataset, self.dictionary.bos, self.dictionary.eos
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
                        pad_idx=self.dictionary.pad,
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


class ProteinDataset(DictDataset):
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
                pad_idx=self.dictionary.pad,
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
                encoder_token_dataset, self.dictionary.bos, self.dictionary.eos
            )
            tgt_dataset = PrependAndAppend(
                encoder_target_dataset, self.dictionary.pad, self.dictionary.pad
            )
            edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
            coord_dataset = FromNumpyDataset(coord_dataset)
            coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
            distance_dataset = DistanceDataset(coord_dataset)
            return {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad,
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
                    tgt_dataset, pad_idx=self.dictionary.pad
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

class CrossDataset(DictDataset):
    def __init__(self, 
                 args, 
                 dictionary, 
                 pocket_dictionary,
                 mole_diffusion_sampler = None,
                 protein_diffusion_sampler = None,
                 atom_diffusion_sampler = None):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.args = Namespace(**args)
        self.seed = self.args.seed
        self.mole_diffusion_sampler = mole_diffusion_sampler
        self.protein_diffusion_sampler = protein_diffusion_sampler
        self.atom_diffusion_sampler = atom_diffusion_sampler
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)
        self.datasets = {}

    def load_lmdb(self, path, split):
        data_path = os.path.join(path, split + ".lmdb")
        self.raw_dataset = LMDBDataset(data_path)
        self.transform(split = split,
                        mole_diffusion_sampler = self.mole_diffusion_sampler,
                        protein_diffusion_sampler = self.protein_diffusion_sampler,
                        atom_diffusion_sampler = self.atom_diffusion_sampler)
    
    def transform(self,
                  split,
                  epoch = 1,
                  mole_diffusion_sampler = None,
                  protein_diffusion_sampler = None,
                  atom_diffusion_sampler = None):
        dataset = self.raw_dataset
        smi_dataset = KeyDataset(dataset, "smi")
        poc_dataset = KeyDataset(dataset, "pocket")
        dataset = ConformerSampleDockingPoseDataset(
            dataset,
            self.args.seed,
            "atoms",
            "coordinates",
            "pocket_atoms",
            "pocket_coordinates",
            "holo_coordinates", # The original ligand coordinates in the complex
            "holo_pocket_coordinates", # holo pocket coordinates = pocket coordinates
            True, # is_train = False, this will make holo coordinates = coordinates
        )
        self.conformer_sample_dataset = dataset

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDockingPoseDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            self.args.max_pocket_atoms,
        )
        dataset = RemoveHydrogenPocketDataset(
            dataset, "atoms", "coordinates", "holo_coordinates", True, True
        )
        # dataset = NormalizeDockingPoseDataset(
        #     dataset,
        #     "coordinates",
        #     "pocket_coordinates",
        #     "center_coordinates",
        # )#Move the whole complex to centralize the protein pocket.
        normalization = False if split in ["train", "train.small"] else True
        apo_dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=normalization)
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates", normalize_coord=normalization)
        src_dataset = KeyDataset(apo_dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        if atom_diffusion_sampler is not None:
            diffused_dataset = DiffusionDataset(
                src_dataset,
                self.args.seed,
                is_train=True,
                diffuser=atom_diffusion_sampler,
            )
            diffused_dataset.set_epoch(epoch)
            src_diffused_dataset = KeyDataset(diffused_dataset, "diffused")
            src_diffused_dataset = PrependAndAppend(
                src_diffused_dataset, self.dictionary.bos, self.dictionary.eos
            )
            diffused_edge_type = EdgeTypeDataset(
                src_diffused_dataset, len(self.dictionary)
            )
            src_score_dataset = KeyDataset(diffused_dataset, "score")
            src_score_dataset = FromNumpyDataset(src_score_dataset)
            src_score_dataset = PrependAndAppend(src_score_dataset, 0.0, 0.0)
            src_norm_dataset = KeyDataset(diffused_dataset, "norm")
            src_norm_dataset = FromNumpyDataset(src_norm_dataset)
            src_norm_dataset = PrependAndAppend(src_norm_dataset, 0.0, 0.0)
            src_time_dataset = KeyDataset(diffused_dataset, "time_steps")
            src_time_dataset = FromNumpyDataset(src_time_dataset)
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos, self.dictionary.eos
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))

        ## Processing coordinates
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        displacement_dataset = DisplacementDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, np.inf, np.inf)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
        displacement_dataset = PrependAndAppend3DDataset(displacement_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos,
            self.pocket_dictionary.eos,
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        displacement_pocket_dataset = DisplacementDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, np.inf, np.inf)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )
        displacement_pocket_dataset = PrependAndAppend3DDataset(
            displacement_pocket_dataset, 0.0
        )
        cross_edgetype_dataset = CrossEdgeTypeDataset(
            src_dataset, src_pocket_dataset, len(self.dictionary)
        )
        holo_dataset = NormalizeDockingPoseDataset(
            dataset,
            "holo_coordinates",
            "holo_pocket_coordinates",
            "holo_center_coordinates",
        )   
        holo_coord_dataset = KeyDataset(holo_dataset, "holo_coordinates")
        holo_coord_dataset = FromNumpyDataset(holo_coord_dataset)
        holo_coord_pocket_dataset = KeyDataset(holo_dataset, "holo_pocket_coordinates")
        holo_coord_pocket_dataset = FromNumpyDataset(holo_coord_pocket_dataset)
        holo_center_coordinates = KeyDataset(holo_dataset, "holo_center_coordinates")
        holo_cross_distance_dataset = CrossDistanceDataset(
            holo_coord_dataset, holo_coord_pocket_dataset
        )
        holo_cross_displacement_dataset = CrossDisplacementDataset(
            holo_coord_dataset, holo_coord_pocket_dataset
        )
        holo_distance_dataset = DistanceDataset(holo_coord_dataset)
        holo_displacement_dataset = DisplacementDataset(holo_coord_dataset)
        ## Add diffusion to protein atom
        if protein_diffusion_sampler is not None:
            pocket_diffused_dataset = DiffusionDataset(
                holo_coord_pocket_dataset,
                self.args.seed,
                is_train=True,
                diffuser=protein_diffusion_sampler,
            )
            pocket_diffused_dataset.set_epoch(epoch)
            coord_pocket_diffused_dataset_nocat = KeyDataset(pocket_diffused_dataset, "diffused")
            distance_pocket_diffused_dataset = DistanceDataset(coord_pocket_diffused_dataset_nocat)
            displacement_pocket_diffused_dataset = DisplacementDataset(coord_pocket_diffused_dataset_nocat)
            coord_pocket_diffused_dataset = PrependAndAppend(coord_pocket_diffused_dataset_nocat, np.inf, np.inf)
            distance_pocket_diffused_dataset = PrependAndAppend2DDataset(distance_pocket_diffused_dataset, 0.0)
            displacement_pocket_diffused_dataset = PrependAndAppend3DDataset(displacement_pocket_diffused_dataset, 0.0)
            pocket_score_dataset = KeyDataset(pocket_diffused_dataset, "score")
            pocket_score_dataset = FromNumpyDataset(pocket_score_dataset)
            pocket_norm_dataset = KeyDataset(pocket_diffused_dataset, "norm")
            pocket_norm_dataset = FromNumpyDataset(pocket_norm_dataset)
            pocket_score_dataset = PrependAndAppend(pocket_score_dataset, 0.0, 0.0)
            pocket_norm_dataset = PrependAndAppend(pocket_norm_dataset, 0.0, 0.0)
            pocket_time_dataset = KeyDataset(pocket_diffused_dataset, "time_steps")
            pocket_time_dataset = FromNumpyDataset(pocket_time_dataset)
        ## Add diffusion noise to the dataset
        if mole_diffusion_sampler is not None:
            mol_diffused = DiffusionDataset(
                holo_coord_dataset,
                self.args.seed,
                is_train=True,
                diffuser=mole_diffusion_sampler,
            )
            mol_diffused.set_epoch(epoch)
            holo_coord_diffused_nocat = KeyDataset(mol_diffused, "diffused")
            holo_distance_diffused = DistanceDataset(holo_coord_diffused_nocat)
            holo_displacement_diffused = DisplacementDataset(holo_coord_diffused_nocat)
            holo_coord_diffused = PrependAndAppend(holo_coord_diffused_nocat, np.inf, np.inf)
            holo_distance_diffused = PrependAndAppend2DDataset(holo_distance_diffused, 0.0)
            holo_displacement_diffused = PrependAndAppend3DDataset(holo_displacement_diffused, 0.0)
            holo_time_dataset = KeyDataset(mol_diffused, "time_steps")
            holo_time_dataset = FromNumpyDataset(holo_time_dataset)
            coord_score_dataset = KeyDataset(mol_diffused, "score")
            coord_trrot_score_dataset = SliceDataset(coord_score_dataset, end=2)
            coord_trrot_score_dataset = FromNumpyDataset(coord_trrot_score_dataset)
            coord_perturb_score_dataset = SliceDataset(coord_score_dataset, start=2)
            coord_perturb_score_dataset = FromNumpyDataset(coord_perturb_score_dataset)
            coord_perturb_score_dataset = PrependAndAppend(coord_perturb_score_dataset, 0.0, 0.0)
            coord_norm_dataset = KeyDataset(mol_diffused, "norm")
            coord_trrot_norm_dataset = SliceDataset(coord_norm_dataset, end=2)
            coord_trrot_norm_dataset = FromNumpyDataset(coord_trrot_norm_dataset)
            coord_perturb_norm_dataset = SliceDataset(coord_norm_dataset, start=2)
            coord_perturb_norm_dataset = FromNumpyDataset(coord_perturb_norm_dataset)
            coord_perturb_norm_dataset = PrependAndAppend(coord_perturb_norm_dataset, 0.0, 0.0)
            
            # ### Debug code
            # diffused_molecular_coord_orig = mol_diffused[0]["diffused"]
            # diffused_molecular_coord = holo_coord_diffused_nocat[0]
            # assert np.allclose(diffused_molecular_coord_orig.cpu().numpy(), diffused_molecular_coord.cpu().numpy()), "The diffused coordinates are not the same"
            # molecular_coord = holo_coord_dataset[0].cpu().numpy()
            # translation_score = mol_diffused[0]["score"][1,:] #rot -> tr -> pert, so  the second one is the translation score
            # translation_score_direct = coord_score_dataset[0][1,:]
            # assert np.allclose(translation_score, translation_score_direct), "The translation score is not the same"
            # disp = diffused_molecular_coord.mean(axis = 0) - molecular_coord.mean(axis = 0)
            # correlation = np.corrcoef(translation_score, disp)
            # print(correlation[0,1])
            # assert correlation[0,1] > 0.99, f"The correlation between the translation score and the displacement is too low: {correlation} "
            # ###

        if mole_diffusion_sampler is not None or protein_diffusion_sampler is not None:
            if mole_diffusion_sampler is None:
                holo_coord_diffused_nocat = holo_coord_dataset
            if protein_diffusion_sampler is None:
                coord_pocket_diffused_dataset_nocat = holo_coord_pocket_dataset
            diffuse_cross_distance_dataset = CrossDistanceDataset(
                holo_coord_diffused_nocat, coord_pocket_diffused_dataset_nocat
            )
            diffuse_cross_displacement_dataset = CrossDisplacementDataset(
                holo_coord_diffused_nocat, coord_pocket_diffused_dataset_nocat
            )
            diffuse_cross_distance_dataset = PrependAndAppend2DDataset(
                diffuse_cross_distance_dataset, 0.0
            )
            diffuse_cross_displacement_dataset = PrependAndAppend3DDataset(
                diffuse_cross_displacement_dataset, 0.0
            )
        if atom_diffusion_sampler is not None:
            diffused_cross_edge_type = CrossEdgeTypeDataset(
                src_diffused_dataset, coord_pocket_diffused_dataset, len(self.dictionary)
            )
        else:
            diffused_cross_edge_type = cross_edgetype_dataset
        holo_coord_dataset = PrependAndAppend(holo_coord_dataset, np.inf, np.inf)
        holo_distance_dataset = PrependAndAppend2DDataset(holo_distance_dataset, 0.0)
        holo_displacement_dataset = PrependAndAppend3DDataset(holo_displacement_dataset, 0.0)
        holo_coord_pocket_dataset = PrependAndAppend(
            holo_coord_pocket_dataset, np.inf, np.inf
        )
        holo_cross_distance_dataset = PrependAndAppend2DDataset(
            holo_cross_distance_dataset, 0.0
        )
        holo_cross_displacement_dataset = PrependAndAppend3DDataset(
            holo_cross_displacement_dataset, 0.0
        )
        holo_center_coordinates = FromNumpyDataset(holo_center_coordinates)

        return_dict = {
                "net_input": {
                    "mol_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad,
                    ),
                    "mol_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "mol_src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=np.inf,
                    ), #This coordinate is the molecule coordinate optimized by rdkit (without pocket atoms)
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ), #Corresponding distance matrix of the mol_src_coord
                    "mol_src_displacement": RightPadDataset3D(
                        displacement_dataset,
                        pad_idx=0,
                    ), #Displacement matrix (n,n,3) of the mol_src_coord
                    "mol_holo_coord": RightPadDatasetCoord(holo_coord_dataset, pad_idx=np.inf),
                    "mol_holo_distance": RightPadDataset2D(
                        holo_distance_dataset, pad_idx=0
                    ),#Holo coordinate and distance should be used as label (true coordinate in complex but normalized by pocket center)
                    "mol_holo_displacement": RightPadDataset3D(
                        holo_displacement_dataset, pad_idx=0
                    ), #Displacement matrix (n,n,3) of the mol_holo_coord
                    "pocket_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad,
                    ),
                    "pocket_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_displacement": RightPadDataset3D(
                        displacement_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=np.inf,
                    ),
                    "pocket_holo_coord": RightPadDatasetCoord(
                        holo_coord_pocket_dataset, pad_idx=np.inf
                    ), #This is the normalized pocket_src_coord 
                    "cross_distance": RightPadDatasetCross2D(
                        holo_cross_distance_dataset, pad_idx=0
                    ),
                    "cross_displacement": RightPadDatasetCross3D(
                        holo_cross_displacement_dataset, pad_idx=0
                    ),
                    "cross_edge_type": RightPadDatasetCross2D(
                        cross_edgetype_dataset, pad_idx=0
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
                "holo_center_coordinates": holo_center_coordinates,
            }
        if atom_diffusion_sampler or protein_diffusion_sampler or mole_diffusion_sampler:
            return_dict["diffused"] = {}
        if atom_diffusion_sampler is not None:
            return_dict['diffused'].update(
                {   "mol_tokens": RightPadDataset(
                        src_diffused_dataset, 
                        pad_idx=self.dictionary.pad
                    ),
                    "mol_edge_type": RightPadDataset2D(
                        diffused_edge_type, pad_idx=0
                    ),
                    "atom_diffuse_time": RightPadDataset(
                        src_time_dataset, pad_idx=0
                    ),
                    "atom_diffuse_score": RightPadDataset(
                        src_score_dataset, pad_idx=0
                    ),
                    "atom_diffuse_norm": RightPadDataset(
                        src_norm_dataset, pad_idx=0
                    ),
                }
            )
        if protein_diffusion_sampler is not None:
            return_dict["diffused"].update(
                {   
                    "pocket_holo_coord": RightPadDatasetCoord(
                        coord_pocket_diffused_dataset, pad_idx=np.inf
                    ),
                    "pocket_distance": RightPadDataset2D(
                        distance_pocket_diffused_dataset, pad_idx=0
                    ),
                    "pocket_displacement": RightPadDataset3D(
                        displacement_pocket_diffused_dataset, pad_idx=0
                    ),
                    "pocket_diffuse_time": pocket_time_dataset,
                    "pocket_diffuse_score": RightPadDatasetCoord(
                        pocket_score_dataset, pad_idx=0
                    ),
                    "pocket_diffuse_norm": RightPadDataset(
                        pocket_norm_dataset, pad_idx=0
                    ),
                }
            )
        if mole_diffusion_sampler is not None:
            return_dict["diffused"].update(
                {   "mol_holo_coord": RightPadDatasetCoord(
                        holo_coord_diffused, pad_idx=np.inf
                    ),
                    "mol_holo_distance": RightPadDataset2D(
                        holo_distance_diffused, pad_idx=0
                    ),
                    "mol_holo_displacement": RightPadDataset3D(
                        holo_displacement_diffused, pad_idx=0
                    ),
                    "mol_diffuse_time": holo_time_dataset,
                    "mol_diffuse_trrot_score": RightPadDatasetCoord(
                        coord_trrot_score_dataset, pad_idx=0
                    ),
                    "mol_diffuse_perturb_score": RightPadDatasetCoord(
                        coord_perturb_score_dataset, pad_idx=0
                    ),
                    "mol_diffuse_trrot_norm": RightPadDataset(
                        coord_trrot_norm_dataset, pad_idx=0
                    ),
                    "mol_diffuse_perturb_norm": RightPadDataset(
                        coord_perturb_norm_dataset, pad_idx=0
                    ),                    
                }
            )
        if mole_diffusion_sampler is not None or protein_diffusion_sampler is not None or atom_diffusion_sampler is not None:
            return_dict["diffused"].update(
                {   "cross_distance": RightPadDatasetCross2D(
                        diffuse_cross_distance_dataset, pad_idx=0
                    ),
                    "cross_displacement": RightPadDatasetCross3D(
                        diffuse_cross_displacement_dataset, pad_idx=0
                    ),
                    "cross_edge_type": RightPadDatasetCross2D(
                        diffused_cross_edge_type, pad_idx=0
                    ),
                }
            )
        nest_dataset = NestedDictionaryDataset(return_dict)
        if split.startswith("train"):
            nest_dataset = EpochShuffleDataset(
                nest_dataset, len(nest_dataset), self.args.seed
            )
        self.datasets[split] = nest_dataset

if __name__ == "__main__":
    from dtmol.utils.dictionary import Dictionary
    from dtmol.dtmol_init import PRETRAIN_FOLDER
    from dtmol.utils.sampling import rmsd as rmsd_import
    from matplotlib import pyplot as plt
    def rmsd(mol1, mol2):
        """
        mol1: [n,3]
        mol2: [n,3]
        """
        return torch.mean(torch.sqrt(torch.sum((mol1 - mol2)**2,dim = -1)))
    ligand_dict = Dictionary.load(f"{PRETRAIN_FOLDER}/unimol_molecule_dict.txt")
    protein_dict = Dictionary.load(f"{PRETRAIN_FOLDER}/unimol_protein_dict.txt")
    protein_path = "/data/unimol_data/protein_ligand_binding_pose_prediction/"
    test_config = {
        "seed": 0,
        "max_seq_len": 1000,
        "max_pocket_atoms": 256,
    }
    pocket_dataset = CrossDataset(test_config,ligand_dict,protein_dict)
    pocket_dataset.load_lmdb(protein_path,"train")
    pocket_dataset.transform("train")

    #3D plot the pocket and ligand
    idx = 2
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    print(pocket_dataset['train'][idx]['pocket_name'])
    print(pocket_dataset['train'][idx]['smi_name'])
    pocket = pocket_dataset['train'][idx]['net_input.pocket_holo_coord']
    ligand = pocket_dataset['train'][idx]['net_input.mol_holo_coord']
    ligand_src = pocket_dataset['train'][idx]['net_input.mol_src_coord']
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(pocket[:, 0], pocket[:, 1], pocket[:, 2], c='pink', marker='o', label = "pocket")
    ax.scatter(ligand[:, 0], ligand[:, 1], ligand[:, 2], c='b', marker='o', label = "ligand")
    ax.scatter(ligand_src[:, 0], ligand_src[:, 1], ligand_src[:, 2], c='purple', marker='o', label = "ligand rdkit")
    print("RMSD of ligand and ligand rdkit: ", rmsd(ligand_src[1:-1], ligand[1:-1]))
    print("RMSD2 of ligand and ligand rdkit: ", rmsd_import(ligand_src.unsqueeze(0), ligand.unsqueeze(0), torch.zeros(1,ligand.shape[0]-1).bool(), torch.zeros(1,1,dtype = torch.bool).unsqueeze(0)[:,:,0]))

    #Test diffuser
    from dtmol.diffusion import RotationSampler, GaussianSampler, TranslationSampler, ChainSampler
    from dtmol.diffusion import GeometricScheduler, PolynomialScheduler, CosineScheduler
    T = 5000
    cos_sch = CosineScheduler(T)
    geo_sch = GeometricScheduler(T)
    poly_sch = PolynomialScheduler(T)
    rot_sampler = RotationSampler(schedular=cos_sch)
    g_sampler = GaussianSampler(schedular = cos_sch)
    g_sampler2 = GaussianSampler(schedular = cos_sch)
    tr_sampler = TranslationSampler(schedular = cos_sch)
    molecule_sampler = ChainSampler(rot_sampler).compose(tr_sampler).compose(g_sampler)
    protein_sampler = ChainSampler(g_sampler2)
    protein_sampler.conjugate(molecule_sampler)
    diffuse_dataset = CrossDataset(args = test_config,
                                   dictionary=ligand_dict,
                                   pocket_dictionary=protein_dict,
                                   mole_diffusion_sampler=molecule_sampler,
                                   protein_diffusion_sampler=protein_sampler)
    diffuse_dataset.load_lmdb(protein_path,"train")
    diffused_pocket = diffuse_dataset['train'][idx]['diffused.pocket_holo_coord']
    diffused_ligand = diffuse_dataset['train'][idx]['diffused.mol_holo_coord']
    assert diffuse_dataset['train'][idx]['diffused.pocket_diffuse_time'] == diffuse_dataset['train'][idx]['diffused.mol_diffuse_time']
    # ax.scatter(diffused_pocket[:, 0], diffused_pocket[:, 1], diffused_pocket[:, 2], c='r', marker='o', label = "diffused pocket")
    ax.scatter(diffused_ligand[:, 0], diffused_ligand[:, 1], diffused_ligand[:, 2], c='g', marker='o', label = "diffused ligand")
    print("RMSD of ligand and diffused ligand: ", rmsd(diffused_ligand[1:-1], ligand[1:-1]))
    plt.legend()

    print(f"Diffuse ligand score shape: {diffuse_dataset['train'][idx]['diffused.mol_diffuse_perturb_score'].shape}")
    print(f"Diffuse ligand coordinate shape: {diffused_ligand.shape}")
    print("""Notice the diffused score and ligand coordinates would have same shape with a length of n_atoms+2! 
The reason is that for the additional dimension due to rotation and translation diffusion
and for the prepend_and_append of <bos> and <eos> tokens""")
    
    print(f"Diffuse pocket score shape: {diffuse_dataset['train'][idx]['diffused.pocket_diffuse_score'].shape}")
    print(f"Diffuse pocket coordinate shape: {diffused_pocket.shape}")
    print("""Notice the diffused score and pocket coordinates would have same shape with a length of n_atoms+2!
We prepend and append tokens to the pocket score, as pocket atom won't have rotation and translation diffuser""")
    
    ### Try to visulaize the molecule using rdkit
    import torch
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdDetermineBonds
    coords_orig = pocket_dataset['train'][idx]['net_input.mol_src_coord'][1:-1]
    coords = diffuse_dataset['train'][idx]['diffused.mol_holo_coord'][1:-1]
    tokens = diffuse_dataset['train'][idx]['net_input.mol_tokens'].numpy()[1:-1]
    smile = pocket_dataset['train'][idx]['smi_name']
    token_to_atom_dict = protein_dict
    def create_molecule(coordinates, tokens, smile, token_to_atom_dict):
        # Initialize an empty editable molecule
        mol = Chem.RWMol()
        template = Chem.MolFromSmiles(smile)
        # Add atoms to molecule
        for token in tokens:
            atom_symbol = token_to_atom_dict[token]
            atom = Chem.Atom(atom_symbol)
            mol.AddAtom(atom)
        
        # Set coordinates
        conf = Chem.Conformer(len(tokens))
        for i, coord in enumerate(coordinates):
            point = rdkit.Geometry.Point3D(coord[0].item(), coord[1].item(), coord[2].item())
            conf.SetAtomPosition(i, point)
        mol.AddConformer(conf)
        # Add bonds to molecule
        rdDetermineBonds.DetermineConnectivity(mol)
        # Add Hydrogens
        mol = Chem.AddHs(mol)
        # decide bond order
        # rdDetermineBonds.DetermineBondOrders(mol,charge=0)
        cm = Chem.RemoveHs(mol)
        osmi = Chem.MolToSmiles(cm)
        print(osmi)
        print(smile)
        # AllChem.EmbedMolecule(mol)


        # Infer bonds
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)  # To ensure the ring information is up-to-date
        AllChem.SanitizeMol(mol)

        # Visualize the molecule
        return mol

    mol = create_molecule(coords, tokens, smile, token_to_atom_dict)
    #save to .mol2 file
    test_folder = "/home/haotiant/Projects/CMU/dtmol/dtmol/test_data/visual_test"
    writer = Chem.SDWriter(os.path.join(test_folder, f"{diffuse_dataset['train'][idx]['pocket_name']}.sdf"))
    writer.write(mol)
    import py3Dmol
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.ipython_3d = True
    def draw_with_spheres(mol):
        v = py3Dmol.view(width=300,height=300)
        IPythonConsole.addMolToView(mol,v)
        v.zoomTo()
        v.setStyle({'sphere':{'radius':0.3},'stick':{'radius':0.2}})
        v.show()
    draw_with_spheres(mol)
    
    
    os.makedirs(test_folder,exist_ok = True)
    with open(os.path.join(test_folder,'coords.npy'), 'wb+') as f:
        np.save(f, coords.numpy())
    with open(os.path.join(test_folder,'tokens.npy'), 'wb+') as f:
        np.save(f, [token_to_atom_dict[t] for t in tokens])
    