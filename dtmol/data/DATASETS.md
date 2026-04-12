# dtMol Dataset Reference

All datasets are stored under \`<data-root>/\` (default: \`/data/dtMol_Project/datasets/\`).

## Dataset Summary

| Dataset | Source URL | Size | License | Raw Format | Auto-Download |
|---------|-----------|------|---------|------------|---------------|
| QM9 | [Figshare](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904) | ~700 MB | CC0 1.0 | SDF + CSV | No (manual) |
| PDBBind | [PDBBind](http://www.pdbbind.org.cn/) | ~2 GB | Academic use | LMDB (pre-processed) | No (manual) |
| ANI-2x | [Zenodo 10108942](https://zenodo.org/records/10108942) | ~7 GB | CC BY 4.0 | HDF5 | Yes |
| SPICE2 | [Zenodo 8222043](https://zenodo.org/records/8222043) | ~3 GB | CC BY 4.0 | HDF5 | Yes |
| Transition1x | [Zenodo 7793881](https://zenodo.org/records/7793881) | ~9 GB | CC BY 4.0 | HDF5 | Yes |
| PDB_apo | [RCSB PDB](https://www.rcsb.org/) | ~100 MB | CC0 1.0 | PDB/mmCIF | Yes |
| MISATO | [Zenodo 7711953](https://zenodo.org/records/7711953) | ~300 MB (QM) / ~133 GB (MD) | CC BY 4.0 | HDF5 | Yes |

## Per-Dataset Details

### QM9

- **Description**: 134k small organic molecules (up to 9 heavy atoms: C, N, O, F) with DFT properties (B3LYP/6-31G(2df,p))
- **Citation**: Ramakrishnan et al., Sci. Data 1, 140022 (2014)
- **Raw location**: \`<root>/QM9/raw/gdb9.sdf\` + \`gdb9.sdf.csv\`
- **Converter**: \`python -m dtmol.data.convert --source qm9 --input <root>/QM9/raw --output <root>/QM9/unified\`
- **PES tier**: C (energies only, no forces)

### PDBBind

- **Description**: Protein-ligand binding pose prediction dataset (~13-19k complexes)
- **Citation**: Wang et al., J. Med. Chem. 47, 2977 (2004)
- **Raw location**: \`<root>/PDBBind/pdbbind.lmdb/\` or \`/data/unimol_data/protein_ligand_binding_pose_prediction/\`
- **Converter**: \`python -m dtmol.data.convert --source pdbbind --input <root>/PDBBind/pdbbind.lmdb --output <root>/PDBBind/unified\`
- **PES tier**: C (no forces)

### ANI-2x

- **Description**: ~10M conformations of small organic molecules with wB97X/6-31G(d) DFT energies and forces
- **Citation**: Devereux et al., J. Chem. Theory Comput. 16, 4192 (2020)
- **Raw location**: \`<root>/ANI-2x/ANI-2x-wB97X-631Gd.h5\`
- **Converter**: \`python -m dtmol.data.convert --source ani2x --input <root>/ANI-2x/ANI-2x-wB97X-631Gd.h5 --output <root>/ANI-2x/unified\`
- **PES tier**: A (energies + forces)
- **Unit conversion**: Hartree -> eV, Hartree/Bohr -> eV/Angstrom

### SPICE2

- **Description**: Drug-like molecules and dimers with DFT properties (wB97M-D3BJ/def2-TZVPPD)
- **Citation**: Eastman et al., Sci. Data 10, 11 (2023)
- **Raw location**: \`<root>/SPICE2/SPICE-2.0.0.hdf5\`
- **Converter**: \`python -m dtmol.data.convert --source spice2 --input <root>/SPICE2/SPICE-2.0.0.hdf5 --output <root>/SPICE2/unified\`
- **PES tier**: A (energies + forces)
- **Unit conversion**: kJ/mol -> eV, kJ/mol/nm -> eV/Angstrom; gradients negated to get forces

### Transition1x (IRC)

- **Description**: Reaction paths / intrinsic reaction coordinate trajectories with DFT energies and forces
- **Citation**: Schreiner et al., Sci. Data 9, 779 (2022)
- **Raw location**: \`<root>/Transition1x/Transition1x.h5\`
- **Converter**: \`python -m dtmol.data.convert --source irc --input <root>/Transition1x/Transition1x.h5 --output <root>/Transition1x/unified\`
- **PES tier**: A (energies + forces)
- **Unit conversion**: Hartree -> eV, Hartree/Bohr -> eV/Angstrom

### PDB_apo

- **Description**: ~100 representative apo (unliganded) protein structures for protein-only pretraining
- **Citation**: Berman et al., Nucleic Acids Res. 28, 235 (2000)
- **Raw location**: \`<root>/PDB_apo/*.pdb\` (or \`*.cif\`)
- **Converter**: \`python -m dtmol.data.convert --source pdb_apo --input <root>/PDB_apo --output <root>/PDB_apo/unified\`
- **PES tier**: C (structural only, no energies or forces)

### MISATO

- **Description**: Protein-ligand MD trajectories with displacement-based force proxies
- **Citation**: Siebenmorgen et al., J. Chem. Inf. Model. 64, 2539 (2024)
- **Raw location**: \`<root>/MISATO/QM_data.hdf5\` (QM portion) or \`MD_data.hdf5\` (full MD)
- **Converter**: \`python -m dtmol.data.convert --source misato --input <root>/MISATO/QM_data.hdf5 --output <root>/MISATO/unified\`
- **PES tier**: B (displacement-based force proxies, not true DFT forces)

## Download Commands

\`\`\`bash
# Check status of all datasets
python -m dtmol.data.download --check

# Download all auto-downloadable datasets (~20 GB without MISATO MD)
python -m dtmol.data.download --all

# Download a single dataset
python -m dtmol.data.download --source ani2x

# Download with MISATO full MD (133 GB, requires confirmation)
python -m dtmol.data.download --all --misato-md

# Use custom PDB ID list for apo download
python -m dtmol.data.download --source pdb_apo --pdb-list my_pdb_ids.txt
\`\`\`
