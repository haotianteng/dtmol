# Unit conversion constants (all processors normalize to eV and eV/Angstrom)
HARTREE_TO_EV = 27.211386245988
KCAL_MOL_TO_EV = 0.04336411153
BOHR_TO_ANGSTROM = 0.529177249

# Atomic number to symbol mapping
ATOMIC_NUMBERS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 26: "Fe", 29: "Cu",
    30: "Zn", 34: "Se", 35: "Br", 53: "I",
}


def write_lmdb(records, db_path, map_size=1099511627776):
    """Write a list of dicts to an LMDB file.

    Args:
        records: iterable of dict, each record to store.
        db_path: str, path to the output .lmdb file.
        map_size: int, LMDB map size in bytes (default 1TB).

    Returns:
        int, number of records written.
    """
    import lmdb
    import pickle
    env = lmdb.open(db_path, map_size=map_size)
    count = 0
    txn = env.begin(write=True)
    for i, data in enumerate(records):
        txn.put(str(i).encode(), pickle.dumps(data))
        count = i + 1
        if count % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    return count
