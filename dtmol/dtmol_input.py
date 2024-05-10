import torch
from torch.utils import data
import dtmol
from dtmol.utils.dictionary import Dictionary
from dtmol.utils.datasets import CrossDataset
from dtmol.utils.dictionary import Dictionary
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dtmol.diffusion import RotationSampler, GaussianSampler, TranslationSampler, ChainSampler, DummySampler
from dtmol.diffusion import GeometricScheduler, PolynomialScheduler, CosineScheduler,LogLinearScheduler

def load_unimol_binding_data(config,
                             data_f,
                             split = ['train','valid','test'],
                             perturbation_mole = True,
                             perturbation_prot = True,
                             trrot = True):
    PRETRAIN_FOLDER = f"{dtmol.__path__[0]}/pretrain_models"
    ligand_dict = Dictionary.load(f"{PRETRAIN_FOLDER}/unimol_molecule_dict.txt")
    protein_dict = Dictionary.load(f"{PRETRAIN_FOLDER}/unimol_protein_dict.txt")
    T = 5000
    cos_sch = CosineScheduler(T)
    # geo_sch = GeometricScheduler(T)
    # poly_sch = PolynomialScheduler(T)
    ll_sch_tr = LogLinearScheduler(T,sigma_min = config['tr_sigma_min'], 
                                   sigma_max = config['tr_sigma_max']) #Parameter value from diffdock tr_sigma_min/max
    ll_sch_rot = LogLinearScheduler(T,sigma_min = config['rot_sigma_min'], 
                                    sigma_max = config['rot_sigma_max'])
    ll_sch_pert = LogLinearScheduler(T,sigma_min = config['pert_mole_sigma_min'], 
                                     sigma_max = config['pert_mole_sigma_max'])
    ll_sch_pert2 = LogLinearScheduler(T,sigma_min = config['pert_prot_sigma_min'], 
                                      sigma_max = config['pert_prot_sigma_max'])
    rot_sampler = RotationSampler(schedular=ll_sch_rot,sde_format = config['rot_sde'])
    g_sampler = GaussianSampler(schedular = ll_sch_pert,sde_format = config['pert_mole_sde'])
    g_sampler2 = GaussianSampler(schedular = ll_sch_pert2,sde_format=config['pert_prot_sde'])
    tr_sampler = TranslationSampler(schedular = ll_sch_tr,sde_format = config['tr_sde'])
    dummy_sampler = DummySampler(schedular = cos_sch,sde_format = config['tr_sde'],system_wise = False)
    dummy_system_sampler = DummySampler(schedular = cos_sch,sde_format = config['tr_sde'],system_wise = True)
    if not trrot:
        rot_sampler = dummy_system_sampler
        tr_sampler = dummy_system_sampler
    if not perturbation_mole:
        g_sampler = dummy_sampler
    if not perturbation_prot:
        g_sampler2 = dummy_sampler
    molecule_sampler = ChainSampler(rot_sampler).compose(tr_sampler).compose(g_sampler)
    protein_sampler = ChainSampler(g_sampler2)
    protein_sampler.conjugate(molecule_sampler) #Sychnronize the time of protein sampler and molecule sampler
    binding_dataset = CrossDataset(config,
                                   ligand_dict,
                                   pocket_dictionary=protein_dict,
                                   mole_diffusion_sampler=molecule_sampler,
                                   protein_diffusion_sampler=protein_sampler)
    for s in split:
        binding_dataset.load_lmdb(data_f,s)
    return binding_dataset

def get_dataloader(dataset,
                   batch_size = 64,
                   split = ['train','valid','test'],
                   device = None,
                   distributed = False):
    loaders = {}
    for s in split:
        ds = dataset[s]
        sampler = DistributedSampler(ds) if distributed else None
        data_loader = DataLoader(ds,
                                collate_fn = ds.collater,
                                batch_size = batch_size,
                                shuffle = (sampler is None),
                                sampler = sampler)
        loaders[s] = DeviceDataLoader(data_loader,device = device)
    return loaders

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dataloader, device = None):
        self.dataloader = dataloader
        if device is None:
            device = self.get_default_device()
        else:
            device = torch.device(device)
        self.device = device
    
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dataloader:
            yield self._to_device(b, self.device)
    
    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)
    
    def _to_device(self,data,device):
        if isinstance(data, (list,tuple)):
            return [self._to_device(x,device) for x in data]
        if isinstance(data, (dict)):
            temp_dict = {}
            for key in data.keys():
                temp_dict[key] = self._to_device(data[key],device)
            return temp_dict
        try:
            return data.to(device, non_blocking=True)
        except AttributeError:
            return data #For non-torch.tensor types
    
    def get_default_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

if __name__ == "__main__":
    # def test_input():
    protein_path = "/data/unimol_data/protein_ligand_binding_pose_prediction/"
    test_config = {
    "seed": 0,
    "max_seq_len": 1000,
    "max_pocket_atoms": 256,
    }
    loader_dict = load_unimol_binding_data(test_config,protein_path)
    loader_dict = get_dataloader(loader_dict,batch_size = 4)
    for batch in loader_dict["train"]:
        print(batch['diffused']['pocket_diffuse_norm'][0])
        print(batch['diffused']['pocket_diffuse_time'])
        print(batch['diffused']['mol_diffuse_time'])
        break
    