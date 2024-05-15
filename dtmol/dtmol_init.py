import dtmol
import os
import logging
from dtmol.utils.so3_op import _pre_compute
logger = logging.getLogger(__name__)
DATA_FOLDER=os.path.join(dtmol.__path__[0],"assets")
PRETRAIN_FOLDER=os.path.join(dtmol.__path__[0],"pretrain_models")
_pre_compute()
PRETRAIN_MOLECULE_URL = "https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt"
PRETRAIN_PROTEIN_URL = "https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt"
if not os.path.exists(os.path.join(PRETRAIN_FOLDER,"unimol_molecule_pretrain.pt")):
    logger.info("Downloading the pretrain molecule model")
    os.system(f"wget -O {PRETRAIN_FOLDER}/unimol_molecule_pretrain.pt https://cmu.box.com/shared/static/8m0l1v0z2x5")
if not os.path.exists(os.path.join(PRETRAIN_FOLDER,"unimol_protein_pretrain.pt")):
    logger.info("Downloading the pretrain protein model")
    os.system(f"wget -O {PRETRAIN_FOLDER}/unimol_protein_pretrain.pt https://cmu.box.com/shared/static/8m0l1v0z2x5")
