# Extract the atoms of the pocket from the protein-ligand complex by 
# searching for the atoms within a certain distance from the ligand.

from biopandas.pdb import PandasPdb

config = {
    'pocket_radius': 10, # Angstrom
}
