# This script trasnfer the PDBBind dataset to LMDB format by extracting the protein pocket and ligand from the PDBBind dataset.
import os
import lmdb
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from biopandas.pdb import PandasPdb
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_pocket(pdb,ligand_coordinates,distance = 10):
    """Extract the atoms of the pocket from the protein-ligand complex by
    searching for the atoms within a certain distance from the ligand, and then extract
    the residues of found atoms.
    Args:
        pdb: str, the path to the PDB file or PDB ID.
        ligand_coordinates: 3xN array, the 3D coordinates of the ligand.
        distance: float, in the unit of Ångstrom, the distance threshold to define the pocket.
                Refernece values: 8.5 Å for water bridges, ~3.5 Å for hydrogen bonds, ~ 4.5 Å for hydrophobic interactions, ~4 Å for van der Waals interactions.

    """
    if len(pdb) == 4:
        pdb_id = pdb
        try:
            ppdb = PandasPdb().fetch_pdb(pdb_id)
        except AttributeError:
            logger.error(f"Failed to fetch the PDB ID {pdb_id}")

    elif os.path.isfile(pdb):
        pdb = os.path.abspath(pdb)
        ppdb = PandasPdb().read_pdb(pdb)
    else:
        raise ValueError("Invalid input for pdb: {}".format(pdb))
    # Extract the atoms within the distance threshold from the ligand
    coordinates = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values
    # Calculate the distance between each atom and the ligand atoms
    mask_protein, mask_ligand = within(coordinates, ligand_coordinates, distance)
    # combine chain and residue number to get unique residue
    ppdb.df['ATOM']["residue_chain"] = ppdb.df['ATOM']["chain_id"] + ppdb.df['ATOM']["residue_number"].astype(str)
    selected_atoms = ppdb.df['ATOM'][mask_protein]
    # Extract the residues of the selected atoms and chains
    residues = selected_atoms['residue_chain'].unique()
    # Extract the atoms of the pocket
    pocket_atoms = ppdb.df['ATOM'][ppdb.df['ATOM']['residue_chain'].isin(residues)]
    return pocket_atoms

def within(src_coords, tgt_coords, distance):
    """Check if the source coordinates are within the distance threshold from the target coordinates.
    Args:
        src_coords: Nx3 array, the source coordinates.
        tgt_coords: Mx3 array, the target coordinates.
        distance: float, the distance threshold.
    Returns:
        mask: NxM boolean array, the mask of whether the source coordinates are within the distance threshold from the target coordinates.
    """
    src_coords = src_coords[:, np.newaxis, :]
    tgt_coords = tgt_coords[np.newaxis, :, :]
    distances = np.linalg.norm(src_coords - tgt_coords, axis=-1)
    mask = distances <= distance
    mask_src = np.any(mask, axis=1)
    mask_tgt = np.any(mask, axis=0)
    return mask_src, mask_tgt

def sdf_to_dataframe(sdf_file):
    """
    Reads an SDF file and returns a pandas DataFrame with atom details.

    :param sdf_file: str, path to the SDF file
    :return: pandas DataFrame with columns 'index', 'atom_name', 'x_coord', 'y_coord', 'z_coord'
    """
    # Create an SDMolSupplier object to read the SDF file
    try:
        supplier = Chem.SDMolSupplier(sdf_file)
    except OSError:
        logger.error(f"Failed to read the SDF file {sdf_file}")
        return None, None

    # Initialize a list to hold atom data
    atom_data = []
    smiles = None
    # Iterate over all molecules in the SDF file
    for mol in supplier:
        if mol is not None:  # Check if the molecule is successfully read
            mol = Chem.RemoveHs(mol)
            try:
                smiles = Chem.MolToSmiles(mol)
            except:
                pass
            mol = AllChem.AddHs(mol, addCoords=True) #This would add potential missing hydroten in the structure
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                element = atom.GetSymbol()
                pos = mol.GetConformer().GetAtomPosition(atom_idx)
                # get the SMILES
                # Add atom data to the list
                atom_data.append([atom_idx, 
                                  element, 
                                  np.float32(pos.x), 
                                  np.float32(pos.y), 
                                  np.float32(pos.z)])
            break # only read the first molecule

    # Create a DataFrame
    columns = ['index', 'atom_name', 'x_coord', 'y_coord', 'z_coord']
    df_atoms = pd.DataFrame(atom_data, columns=columns)
    print(smiles)
    return df_atoms, smiles

def process(pdbbind_dir):
    collection = []
    iterator = tqdm(os.listdir(pdbbind_dir), desc="Processing PDBBind")
    for pdb_id in iterator:
        iterator.set_description(desc = f"Processing {pdb_id}")
        pdb_f = os.path.join(pdbbind_dir, pdb_id, pdb_id + "_protein_processed.pdb")
        ligands_f = os.path.join(pdbbind_dir, pdb_id, pdb_id + "_ligand.sdf")
        ligand_df,smiles = sdf_to_dataframe(ligands_f)
        if ligand_df is None:
            continue
        pocket_atoms = extract_pocket(pdb_f, ligand_df[['x_coord', 'y_coord', 'z_coord']].values)
        current = {"atoms":list(ligand_df['atom_name']), 
                   "coordinates":ligand_df[['x_coord', 'y_coord', 'z_coord']].values,
                   "smi":smiles,
                   "pocket_atoms":list(pocket_atoms['atom_name']),
                   "pocket_coordinates":pocket_atoms[['x_coord', 'y_coord', 'z_coord']].values,
                   "residue":list(pocket_atoms['residue_chain']),
                   "config":{"pocket_radius":10},
                   "pdb_id":pdb_id}
        collection.append(current)
    return collection

if __name__ == "__main__":
    #%% extract pocket test
    pdb_id = "1a07"
    ligands_f = "../test_data/1a07/1a07_ligand.sdf"
    ligand_df,smi = sdf_to_dataframe(ligands_f)
    ligand_coordinates = ligand_df[['x_coord', 'y_coord', 'z_coord']].values
    pocket_atoms = extract_pocket(pdb_id, ligand_coordinates)
    #save to pocket.pdb
    pocket_pdb = PandasPdb()
    pocket_pdb.df['ATOM'] = pocket_atoms
    pocket_pdb.to_pdb(path = "../test_data/1a07/pocket.pdb", records=['ATOM'])

    # transfer to LMDB
    pdbbind_dir = "/data/PDBBind/PDBBind_processed/"
    collection = process(pdbbind_dir)
    # save to LMDB
    db_path = "/data/PDBBind/pdbbind.lmdb"
    env = lmdb.open(db_path, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for i, data in enumerate(collection):
            txn.put(str(i).encode(), pickle.dumps(data))