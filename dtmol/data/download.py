"""Dataset download script for dtMol project.

Usage:
    python -m dtmol.data.download --check --data-root /data/dtMol_Project/datasets
    python -m dtmol.data.download --all --data-root /data/dtMol_Project/datasets
    python -m dtmol.data.download --source ani2x --data-root /data/dtMol_Project/datasets
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# Curated list of ~100 representative apo PDB IDs for protein-only pretraining
_PDB_APO_IDS: List[str] = [
    "1A2P", "1A3N", "1AKE", "1BNA", "1C8C", "1CRN", "1EHZ", "1F88", "1FAT",
    "1GFL", "1GZM", "1HHO", "1HSG", "1IGT", "1J4N", "1K4C", "1L2Y", "1LMB",
    "1MBO", "1NKZ", "1OMP", "1PGB", "1QYS", "1RHD", "1STP", "1TEN", "1TIM",
    "1UBQ", "1VCB", "1W0T", "1XMK", "1YCC", "1ZAA", "2AIT", "2BBM", "2CI2",
    "2CTC", "2DHB", "2DRI", "2FDN", "2GBP", "2HBS", "2HMQ", "2IGD", "2LZM",
    "2MBW", "2NRL", "2OCA", "2PDD", "2POR", "2RN2", "2SNS", "2TRX", "2WRP",
    "3BLM", "3CLN", "3DFR", "3EBX", "3FIB", "3GRS", "3HHB", "3HVP", "3LYZ",
    "3PGK", "3PTE", "3RN3", "3SDH", "3TMS", "4CPA", "4DFR", "4ENL", "4FXN",
    "4GCR", "4HHB", "4INS", "4LYZ", "4MDH", "4PEP", "4PTI", "4TMS", "5CYT",
    "5CPA", "5HVP", "5LYZ", "5PTI", "5RSA", "5TIM", "6LYZ", "6PTI", "6RSA",
    "6TIM", "7AHL", "7CAT", "7RSA", "7TIM", "8CAT", "8TIM", "9PAP", "9WGA",
    "1BPI",
]


def _zenodo_file_url(record_id: int, filename: str) -> str:
    return f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"


# Each dataset entry:
#   dir_name: subdirectory under data_root
#   detect: function(data_root) -> bool  (is it already present?)
#   size_hint: approximate download size string
#   auto: whether it can be auto-downloaded
#   download: function(data_root, **kwargs) -> None
#   license, citation, source_url, raw_format, converter_cmd: for DATASETS.md

_DATASETS: Dict[str, Dict[str, Any]] = {}


def _register(name: str, entry: Dict[str, Any]) -> None:
    _DATASETS[name] = entry


# --- QM9 (detect only) ---
_register("qm9", {
    "dir_name": "QM9",
    "detect": lambda root: (Path(root) / "QM9" / "raw" / "gdb9.sdf").exists(),
    "size_hint": "~700 MB",
    "auto": False,
    "download": None,
    "license": "CC0 1.0",
    "citation": "Ramakrishnan et al., Sci. Data 1, 140022 (2014)",
    "source_url": "https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904",
    "raw_format": "SDF + CSV",
    "converter_cmd": "python -m dtmol.data.convert --source qm9 --input <root>/QM9/raw --output <root>/QM9/unified",
})

# --- PDBBind (detect only) ---
_register("pdbbind", {
    "dir_name": "PDBBind",
    "detect": lambda root: (Path(root) / "PDBBind" / "pdbbind.lmdb").is_dir(),
    "size_hint": "~2 GB",
    "auto": False,
    "download": None,
    "license": "Academic use",
    "citation": "Wang et al., J. Med. Chem. 47, 2977 (2004)",
    "source_url": "http://www.pdbbind.org.cn/",
    "raw_format": "LMDB (pre-processed)",
    "converter_cmd": "python -m dtmol.data.convert --source pdbbind --input <root>/PDBBind/pdbbind.lmdb --output <root>/PDBBind/unified",
})


# --- Helper: download with resume + progress ---
def _download_file(url: str, dest: Path, desc: Optional[str] = None) -> None:
    """Download a file with resume support and progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    existing_size = tmp.stat().st_size if tmp.exists() else 0
    headers: Dict[str, str] = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    resp = requests.get(url, headers=headers, stream=True, timeout=60)

    if resp.status_code == 416:
        # Range not satisfiable — file already complete
        tmp.rename(dest)
        return

    resp.raise_for_status()

    total = resp.headers.get("Content-Length")
    total_size = int(total) + existing_size if total else None

    mode = "ab" if existing_size and resp.status_code == 206 else "wb"
    if mode == "wb":
        existing_size = 0

    with open(tmp, mode) as f, tqdm(
        total=total_size,
        initial=existing_size,
        unit="B",
        unit_scale=True,
        desc=desc or dest.name,
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    tmp.rename(dest)


def _download_and_extract_gz(url: str, dest: Path, desc: Optional[str] = None) -> None:
    """Download a .gz file and decompress it."""
    gz_path = dest.with_suffix(dest.suffix + ".gz")
    _download_file(url, gz_path, desc=desc)
    with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()


# --- ANI-2x ---
def _download_ani2x(root: str, **kwargs: Any) -> None:
    dest_dir = Path(root) / "ANI-2x"
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Check if HDF5 already extracted (may be in a subdirectory from tarball)
    h5_files = list(dest_dir.rglob("*.h5"))
    if h5_files:
        print(f"  Already exists: {h5_files[0]}")
        return
    # ANI-2x on Zenodo record 10108942 — distributed as tar.gz
    url = _zenodo_file_url(10108942, "ANI-2x-wB97X-631Gd.tar.gz")
    tarball = dest_dir / "ANI-2x-wB97X-631Gd.tar.gz"
    _download_file(url, tarball, desc="ANI-2x tar.gz")
    print("  Extracting ANI-2x tar.gz ...")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(path=dest_dir)
    tarball.unlink()
    print("  Extraction complete.")


_register("ani2x", {
    "dir_name": "ANI-2x",
    "detect": lambda root: any((Path(root) / "ANI-2x").rglob("*.h5")),
    "size_hint": "~3.7 GB (compressed)",
    "auto": True,
    "download": _download_ani2x,
    "license": "CC BY 4.0",
    "citation": "Devereux et al., J. Chem. Theory Comput. 16, 4192 (2020)",
    "source_url": "https://zenodo.org/records/10108942",
    "raw_format": "HDF5",
    "converter_cmd": "python -m dtmol.data.convert --source ani2x --input <root>/ANI-2x/ANI-2x-wB97X-631Gd.h5 --output <root>/ANI-2x/unified",
})

# --- SPICE2 ---
def _download_spice2(root: str, **kwargs: Any) -> None:
    dest_dir = Path(root) / "SPICE2"
    dest_dir.mkdir(parents=True, exist_ok=True)
    # SPICE 1.1.4 on Zenodo record 8222043 (largest available HDF5 release)
    url = _zenodo_file_url(8222043, "SPICE-1.1.4.hdf5")
    dest = dest_dir / "SPICE-1.1.4.hdf5"
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    _download_file(url, dest, desc="SPICE HDF5")


_register("spice2", {
    "dir_name": "SPICE2",
    "detect": lambda root: any((Path(root) / "SPICE2").glob("*.hdf5")) or any((Path(root) / "SPICE2").glob("*.h5")),
    "size_hint": "~16 GB",
    "auto": True,
    "download": _download_spice2,
    "license": "CC BY 4.0",
    "citation": "Eastman et al., Sci. Data 10, 11 (2023)",
    "source_url": "https://zenodo.org/records/8222043",
    "raw_format": "HDF5",
    "converter_cmd": "python -m dtmol.data.convert --source spice2 --input <root>/SPICE2/SPICE-1.1.4.hdf5 --output <root>/SPICE2/unified",
})

# --- Transition1x ---
def _download_transition1x(root: str, **kwargs: Any) -> None:
    dest_dir = Path(root) / "Transition1x"
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Transition1x on figshare article 19614657
    url = "https://ndownloader.figshare.com/files/36035789"
    dest = dest_dir / "Transition1x.h5"
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    _download_file(url, dest, desc="Transition1x HDF5")


_register("transition1x", {
    "dir_name": "Transition1x",
    "detect": lambda root: any((Path(root) / "Transition1x").glob("*.h5")),
    "size_hint": "~9 GB",
    "auto": True,
    "download": _download_transition1x,
    "license": "CC BY 4.0",
    "citation": "Schreiner et al., Sci. Data 9, 779 (2022)",
    "source_url": "https://figshare.com/articles/dataset/Transition1x/19614657",
    "raw_format": "HDF5",
    "converter_cmd": "python -m dtmol.data.convert --source irc --input <root>/Transition1x/Transition1x.h5 --output <root>/Transition1x/unified",
})

# --- PDB apo ---
def _download_pdb_apo(root: str, pdb_list: Optional[str] = None, **kwargs: Any) -> None:
    dest_dir = Path(root) / "PDB_apo"
    dest_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = _PDB_APO_IDS
    if pdb_list:
        with open(pdb_list) as f:
            pdb_ids = [line.strip().upper() for line in f if line.strip()]

    for pdb_id in tqdm(pdb_ids, desc="PDB apo structures"):
        pdb_lower = pdb_id.lower()
        dest = dest_dir / f"{pdb_lower}.pdb"
        if dest.exists():
            continue
        url = f"https://files.rcsb.org/download/{pdb_lower}.pdb"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_text(resp.text)
        except requests.RequestException as e:
            # Try mmCIF as fallback
            cif_url = f"https://files.rcsb.org/download/{pdb_lower}.cif"
            try:
                resp = requests.get(cif_url, timeout=30)
                resp.raise_for_status()
                cif_dest = dest_dir / f"{pdb_lower}.cif"
                cif_dest.write_text(resp.text)
            except requests.RequestException:
                print(f"  WARNING: Could not download {pdb_id}: {e}")


_register("pdb_apo", {
    "dir_name": "PDB_apo",
    "detect": lambda root: (Path(root) / "PDB_apo").is_dir() and len(list((Path(root) / "PDB_apo").glob("*.pdb"))) > 10,
    "size_hint": "~100 MB",
    "auto": True,
    "download": _download_pdb_apo,
    "license": "CC0 1.0 (PDB)",
    "citation": "Berman et al., Nucleic Acids Res. 28, 235 (2000)",
    "source_url": "https://www.rcsb.org/",
    "raw_format": "PDB/mmCIF",
    "converter_cmd": "python -m dtmol.data.convert --source pdb_apo --input <root>/PDB_apo --output <root>/PDB_apo/unified",
})

# --- MISATO ---
def _download_misato(root: str, misato_md: bool = False, **kwargs: Any) -> None:
    dest_dir = Path(root) / "MISATO"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # QM portion (~300MB)
    qm_url = _zenodo_file_url(7711953, "QM.hdf5")
    qm_dest = dest_dir / "QM.hdf5"
    if not qm_dest.exists():
        _download_file(qm_url, qm_dest, desc="MISATO QM")
    else:
        print(f"  Already exists: {qm_dest}")

    if misato_md:
        confirm = input(
            "MISATO MD trajectory is ~133 GB. Continue? [y/N] "
        ).strip().lower()
        if confirm != "y":
            print("  Skipping MISATO MD download.")
            return
        md_url = _zenodo_file_url(7711953, "MD.hdf5")
        md_dest = dest_dir / "MD.hdf5"
        if not md_dest.exists():
            _download_file(md_url, md_dest, desc="MISATO MD")
        else:
            print(f"  Already exists: {md_dest}")


_register("misato", {
    "dir_name": "MISATO",
    "detect": lambda root: any((Path(root) / "MISATO").glob("*.hdf5")) or any((Path(root) / "MISATO").glob("*.h5")),
    "size_hint": "~300 MB (QM) / ~133 GB (MD)",
    "auto": True,
    "download": _download_misato,
    "license": "CC BY 4.0",
    "citation": "Siebenmorgen et al., J. Chem. Inf. Model. 64, 2539 (2024)",
    "source_url": "https://zenodo.org/records/7711953",
    "raw_format": "HDF5",
    "converter_cmd": "python -m dtmol.data.convert --source misato --input <root>/MISATO/QM.hdf5 --output <root>/MISATO/unified",
})


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def _get_dir_size(path: Path) -> str:
    """Get human-readable size of a directory."""
    if not path.exists():
        return "-"
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    size = float(total_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def check_datasets(data_root: str) -> None:
    """Print status table of all datasets."""
    print(f"\nDataset status in: {data_root}\n")
    print(f"{'Dataset':<15} {'Status':<10} {'Directory':<30} {'Size':<15} {'Auto-DL'}")
    print("-" * 80)
    for name, info in _DATASETS.items():
        dir_path = Path(data_root) / info["dir_name"]
        found = info["detect"](data_root)
        status = "FOUND" if found else "MISSING"
        size = _get_dir_size(dir_path) if found else "-"
        auto = "Yes" if info["auto"] else "No (manual)"
        print(f"{name:<15} {status:<10} {info['dir_name']:<30} {size:<15} {auto}")
    print()


def download_source(
    data_root: str,
    source: str,
    pdb_list: Optional[str] = None,
    misato_md: bool = False,
) -> None:
    """Download a single dataset by name."""
    if source not in _DATASETS:
        print(f"Unknown source: {source}")
        print(f"Available: {', '.join(_DATASETS.keys())}")
        sys.exit(1)

    info = _DATASETS[source]
    if not info["auto"]:
        print(f"{source}: Not auto-downloadable. Already present: {info['detect'](data_root)}")
        return

    if info["detect"](data_root):
        print(f"{source}: FOUND — skipping download")
        return

    print(f"Downloading {source} ({info['size_hint']}) ...")
    info["download"](data_root, pdb_list=pdb_list, misato_md=misato_md)
    print(f"  Done: {source}")


def download_all(
    data_root: str,
    pdb_list: Optional[str] = None,
    misato_md: bool = False,
) -> None:
    """Download all auto-downloadable datasets."""
    for name, info in _DATASETS.items():
        if info["detect"](data_root):
            print(f"{name}: FOUND — skipping")
            continue
        if not info["auto"]:
            print(f"{name}: Not auto-downloadable — skipping")
            continue
        print(f"\nDownloading {name} ({info['size_hint']}) ...")
        info["download"](data_root, pdb_list=pdb_list, misato_md=misato_md)
        print(f"  Done: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and manage dtMol project datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m dtmol.data.download --check
  python -m dtmol.data.download --source ani2x
  python -m dtmol.data.download --all
  python -m dtmol.data.download --all --misato-md
""",
    )
    parser.add_argument(
        "--data-root",
        default="/data/dtMol_Project/datasets",
        help="Root directory for all datasets (default: /data/dtMol_Project/datasets)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=list(_DATASETS.keys()),
        help="Download a single dataset by name",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help="Download all auto-downloadable datasets",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Scan data-root and print FOUND/MISSING status for all datasets",
    )
    parser.add_argument(
        "--misato-md",
        action="store_true",
        help="Include MISATO MD trajectory (~133GB) — requires confirmation",
    )
    parser.add_argument(
        "--pdb-list",
        type=str,
        default=None,
        help="File with PDB IDs for apo download (one per line), overrides built-in list",
    )

    args = parser.parse_args()

    if args.check:
        check_datasets(args.data_root)
        return

    if args.source:
        download_source(args.data_root, args.source, args.pdb_list, args.misato_md)
        return

    if args.download_all:
        download_all(args.data_root, args.pdb_list, args.misato_md)
        return

    # No action specified — show help
    parser.print_help()


if __name__ == "__main__":
    main()
