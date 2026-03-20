"""Download scripts for all supported datasets.

Usage:
    # Download and process a single dataset
    python -m dtmol.data_op.download --dataset qm9 --output_dir /data/dtmol

    # Download and process all auto-downloadable datasets
    python -m dtmol.data_op.download --dataset all --output_dir /data/dtmol

Datasets that require manual download (PDBBind, CrossDocked2020) will print
instructions instead of downloading automatically.
"""
import os
import sys
import argparse
import subprocess
import logging
import tarfile
import shutil

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Download URLs and metadata
# ============================================================================

DATASET_INFO = {
    "qm9": {
        "description": "QM9: ~134k small organic molecules with DFT properties",
        "auto_download": True,
        "size": "~5 MB (raw), auto-processed by torch_geometric",
        "method": "torch_geometric",
    },
    "ani1x": {
        "description": "ANI-1x: ~5.5M conformations with DFT energies and forces",
        "auto_download": True,
        "size": "~5.2 GB",
        "url": "https://figshare.com/ndownloader/files/18112775",
        "filename": "ani1x-release.h5",
        "method": "wget",
    },
    "md17": {
        "description": "MD17: Ab-initio MD trajectories for 10 small molecules",
        "auto_download": True,
        "size": "~100-300 MB per molecule",
        "method": "torch_geometric",
        "molecules": [
            "aspirin", "benzene", "ethanol", "malonaldehyde", "naphthalene",
            "salicylic_acid", "toluene", "uracil",
        ],
    },
    "rmd17": {
        "description": "Revised MD17: Recomputed with tighter DFT convergence",
        "auto_download": True,
        "size": "~1 GB total",
        "url": "https://figshare.com/ndownloader/articles/12672038/versions/2",
        "filename": "rmd17.tar.bz2",
        "method": "wget",
    },
    "crossdocked": {
        "description": "CrossDocked2020: ~22.5M docked protein-ligand poses",
        "auto_download": False,
        "size": "~50 GB compressed",
        "urls": {
            "data": "http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3.tgz",
            "types": "http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3_types.tgz",
        },
        "method": "manual",
    },
    "pdbbind": {
        "description": "PDBBind: Protein-ligand complexes with binding affinities",
        "auto_download": False,
        "size": "~2-20 GB depending on subset",
        "method": "manual",
        "registration_url": "http://www.pdbbind.org.cn/",
    },
}

# ============================================================================
# Download helpers
# ============================================================================

def _download_wget(url, output_path, filename=None):
    """Download a file using wget."""
    os.makedirs(output_path, exist_ok=True)
    if filename:
        dest = os.path.join(output_path, filename)
    else:
        dest = output_path
    if filename and os.path.exists(dest):
        logger.info(f"File already exists: {dest}, skipping download.")
        return dest
    logger.info(f"Downloading {url} -> {dest}")
    cmd = ["wget", "-q", "--show-progress", "-O", dest, url]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        # wget not available, fall back to Python
        logger.info("wget not found, falling back to urllib...")
        import urllib.request
        urllib.request.urlretrieve(url, dest)
    return dest


def _extract_tar(archive_path, output_dir, mode="r:bz2"):
    """Extract a tar archive."""
    logger.info(f"Extracting {archive_path} -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    with tarfile.open(archive_path, mode) as tar:
        tar.extractall(path=output_dir)


# ============================================================================
# Per-dataset download + process functions
# ============================================================================

def download_qm9(output_dir):
    """Download QM9 via torch_geometric and convert to LMDB."""
    raw_dir = os.path.join(output_dir, "qm9", "raw")
    lmdb_dir = os.path.join(output_dir, "qm9", "lmdb")
    os.makedirs(raw_dir, exist_ok=True)

    logger.info("Downloading QM9 via torch_geometric (auto-download)...")
    try:
        from torch_geometric.datasets import QM9
        QM9(root=raw_dir)  # triggers download + processing
    except ImportError:
        logger.error("torch_geometric is required for QM9: pip install torch_geometric")
        return

    logger.info("Converting QM9 to LMDB...")
    from dtmol.data_op.process_qm9 import process_qm9
    process_qm9(raw_dir, lmdb_dir)
    logger.info(f"QM9 ready at {lmdb_dir}")


def download_ani1x(output_dir):
    """Download ANI-1x HDF5 and convert to LMDB."""
    info = DATASET_INFO["ani1x"]
    raw_dir = os.path.join(output_dir, "ani1x", "raw")
    lmdb_dir = os.path.join(output_dir, "ani1x", "lmdb")

    h5_path = _download_wget(info["url"], raw_dir, info["filename"])

    logger.info("Converting ANI-1x to LMDB (this may take a while for 5.5M conformations)...")
    from dtmol.data_op.process_ani1x import process_ani1x
    process_ani1x(h5_path, lmdb_dir)
    logger.info(f"ANI-1x ready at {lmdb_dir}")


def download_md17(output_dir):
    """Download MD17 via torch_geometric and convert to LMDB."""
    molecules = DATASET_INFO["md17"]["molecules"]

    try:
        from torch_geometric.datasets import MD17
    except ImportError:
        logger.error("torch_geometric is required for MD17: pip install torch_geometric")
        return

    for mol_name in molecules:
        logger.info(f"Downloading MD17/{mol_name} via torch_geometric...")
        raw_dir = os.path.join(output_dir, "md17", "raw", mol_name)
        lmdb_dir = os.path.join(output_dir, "md17", "lmdb", mol_name)
        os.makedirs(raw_dir, exist_ok=True)

        # torch_geometric downloads the npz file
        ds = MD17(root=raw_dir, name=mol_name)

        # Find the downloaded npz file
        npz_files = [f for f in os.listdir(os.path.join(raw_dir, "raw")) if f.endswith(".npz")]
        if not npz_files:
            logger.warning(f"No npz file found for {mol_name}, skipping.")
            continue

        npz_path = os.path.join(raw_dir, "raw", npz_files[0])
        logger.info(f"Converting MD17/{mol_name} to LMDB...")
        from dtmol.data_op.process_md17 import process_md17
        process_md17(npz_path, lmdb_dir, molecule_name=mol_name)

    logger.info(f"MD17 ready at {os.path.join(output_dir, 'md17', 'lmdb')}")


def download_rmd17(output_dir):
    """Download rMD17 from Figshare and convert to LMDB."""
    info = DATASET_INFO["rmd17"]
    raw_dir = os.path.join(output_dir, "rmd17", "raw")
    lmdb_dir = os.path.join(output_dir, "rmd17", "lmdb")

    archive_path = _download_wget(info["url"], raw_dir, info["filename"])

    # Extract the tar.bz2 archive
    npz_dir = os.path.join(raw_dir, "npz")
    if not os.path.isdir(npz_dir) or not os.listdir(npz_dir):
        _extract_tar(archive_path, raw_dir)
        # rMD17 extracts into a subdirectory, find the npz files
        for root, dirs, files in os.walk(raw_dir):
            npz_files = [f for f in files if f.endswith(".npz")]
            if npz_files:
                npz_dir = root
                break

    logger.info(f"Converting rMD17 to LMDB from {npz_dir}...")
    from dtmol.data_op.process_md17 import process_md17_directory
    process_md17_directory(npz_dir, lmdb_dir, revised=True)
    logger.info(f"rMD17 ready at {lmdb_dir}")


def print_manual_instructions(dataset_name):
    """Print download instructions for datasets that require manual download."""
    info = DATASET_INFO[dataset_name]
    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()}: Manual download required")
    print(f"{'='*70}")
    print(f"  {info['description']}")
    print(f"  Size: {info['size']}")

    if dataset_name == "pdbbind":
        print(f"""
  PDBBind requires free registration:
    1. Go to {info['registration_url']}
    2. Create an account and log in
    3. Download the refined/general set
    4. Extract to a directory, e.g., /data/PDBBind/PDBBind_processed/
    5. Run:
       python -m dtmol.data_op.process_pdbbind
       (edit the script to set pdbbind_dir and db_path)
""")
    elif dataset_name == "crossdocked":
        urls = info["urls"]
        print(f"""
  Download the following files:
    wget {urls['data']}
    wget {urls['types']}

  Then extract and process:
    tar xzf CrossDocked2020_v1.3.tgz -C /data/CrossDocked2020/
    tar xzf CrossDocked2020_v1.3_types.tgz -C /data/CrossDocked2020/

    python -m dtmol.data_op.process_crossdocked \\
        --crossdocked_dir /data/CrossDocked2020/ \\
        --types_dir /data/CrossDocked2020/split_by_name/ \\
        --output_dir /data/crossdocked/lmdb
""")
    print(f"{'='*70}\n")


# ============================================================================
# Main dispatcher
# ============================================================================

DOWNLOAD_FUNCTIONS = {
    "qm9": download_qm9,
    "ani1x": download_ani1x,
    "md17": download_md17,
    "rmd17": download_rmd17,
}

AUTO_DOWNLOADABLE = list(DOWNLOAD_FUNCTIONS.keys())
MANUAL_ONLY = ["crossdocked", "pdbbind"]


def download_dataset(dataset_name, output_dir):
    """Download and process a single dataset."""
    if dataset_name in DOWNLOAD_FUNCTIONS:
        DOWNLOAD_FUNCTIONS[dataset_name](output_dir)
    elif dataset_name in MANUAL_ONLY:
        print_manual_instructions(dataset_name)
    else:
        available = list(DATASET_INFO.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")


def download_all(output_dir):
    """Download all auto-downloadable datasets and print instructions for manual ones."""
    for name in AUTO_DOWNLOADABLE:
        logger.info(f"\n{'='*40} {name.upper()} {'='*40}")
        try:
            download_dataset(name, output_dir)
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            continue

    for name in MANUAL_ONLY:
        print_manual_instructions(name)


def list_datasets():
    """Print information about all available datasets."""
    print(f"\n{'Dataset':<15} {'Auto-DL':<10} {'Size':<25} Description")
    print("-" * 90)
    for name, info in DATASET_INFO.items():
        auto = "Yes" if info["auto_download"] else "No"
        print(f"{name:<15} {auto:<10} {info['size']:<25} {info['description']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process datasets for dtmol training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python -m dtmol.data_op.download --list

  # Download and process QM9
  python -m dtmol.data_op.download --dataset qm9 --output_dir /data/dtmol

  # Download all auto-downloadable datasets
  python -m dtmol.data_op.download --dataset all --output_dir /data/dtmol

  # Get instructions for manual-download datasets
  python -m dtmol.data_op.download --dataset pdbbind
        """,
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset to download: qm9, ani1x, md17, rmd17, crossdocked, pdbbind, or 'all'")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Base output directory (default: ./data)")
    parser.add_argument("--list", action="store_true",
                        help="List all available datasets and exit")
    args = parser.parse_args()

    if args.list or args.dataset is None:
        list_datasets()
        sys.exit(0)

    if args.dataset == "all":
        download_all(args.output_dir)
    else:
        download_dataset(args.dataset, args.output_dir)
