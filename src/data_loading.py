"""
src/data_loading.py
===================
Carga el dataset usando los splits oficiales _train.csv / _test.csv.

Estructura por proteína en /CANCERDATASET/:
  {protein}.h5          → fingerprints ap+mg+tt (train+test combinados)
  {protein}_train.csv   → split oficial de entrenamiento (col 0 = label)
  {protein}_test.csv    → split oficial de test           (col 0 = label)

El .h5 concatena train+test en el mismo orden que los CSVs.
Cortamos por len(train_csv): primeras N filas = train, resto = test.

Archivos ignorados:
  pubchem_neg_sample.*  → negativos externos, distribución diferente al lab
  abbr.csv              → metadatos reemplazados por PROTEIN_METADATA
"""
import logging
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy import sparse
from typing import Dict, List, Optional

from src.config import DATA_DIR, FINGERPRINT_SIZE, PROTEIN_METADATA, RANDOM_STATE

logger = logging.getLogger(__name__)
_SKIP = {"pubchem_neg_sample"}


def _read_h5(path: Path) -> Dict:
    """Lee .h5 → sparse float32 + labels int32. float32 requerido por LightGBM."""
    with h5py.File(path, "r") as hf:
        ids = hf["chembl_id"][()].astype(str)
        def sp(key):
            g = hf[key]; ptr = g["indptr"][()]
            return sparse.csr_matrix(
                (g["data"][()], g["indices"][()], ptr),
                shape=(len(ptr)-1, FINGERPRINT_SIZE))
        X = sparse.hstack([sp("ap"), sp("mg"), sp("tt")], format="csr").astype(np.float32)
        y = hf["label"][()].astype(np.int32)
    return {"ids": ids, "X": X, "y": y}


def _fp_slices():
    return {"ap": (0, FINGERPRINT_SIZE),
            "mg": (FINGERPRINT_SIZE, 2*FINGERPRINT_SIZE),
            "tt": (2*FINGERPRINT_SIZE, 3*FINGERPRINT_SIZE)}


def _random_split(X, y, ids):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, ids, test_size=0.20, stratify=y, random_state=RANDOM_STATE)


def load_protein(name: str, data_dir: Optional[Path] = None) -> Dict:
    """Carga proteína con split oficial (o fallback aleatorio si los CSVs faltan)."""
    dd = Path(data_dir) if data_dir else DATA_DIR
    h5d = _read_h5(dd / f"{name}.h5")
    X, y, ids = h5d["X"], h5d["y"], h5d["ids"]

    trn_path = dd / f"{name}_train.csv"
    tst_path = dd / f"{name}_test.csv"

    if trn_path.exists() and tst_path.exists():
        n_tr = len(pd.read_csv(trn_path, header=None))
        n_te = len(pd.read_csv(tst_path, header=None))
        if len(y) == n_tr + n_te:
            X_tr, X_te = X[:n_tr], X[n_tr:]
            y_tr, y_te = y[:n_tr], y[n_tr:]
            ids_tr, ids_te = ids[:n_tr], ids[n_tr:]
            source = "official"
        else:
            logger.warning(f"[{name}] tamaños no coinciden → split aleatorio")
            X_tr, X_te, y_tr, y_te, ids_tr, ids_te = _random_split(X, y, ids)
            source = "random_fallback"
    else:
        logger.warning(f"[{name}] CSVs no encontrados → split aleatorio")
        X_tr, X_te, y_tr, y_te, ids_tr, ids_te = _random_split(X, y, ids)
        source = "random_fallback"

    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    ratio = n_neg / n_pos if n_pos > 0 else float("inf")
    meta  = PROTEIN_METADATA.get(name, {
        "full_name": name, "location": "unknown",
        "family": "unknown", "uniprot": "N/A", "relevance": ""})

    logger.info(f"  {name:15s} | {source} | train={len(y_tr)} "
                f"({n_pos}+ / {n_neg}-) | test={len(y_te)} | {meta['location']}")

    return {"protein": name,
            "X_train": X_tr,   "X_test":  X_te,
            "y_train": y_tr,   "y_test":  y_te,
            "ids_train": ids_tr, "ids_test": ids_te,
            "fp_slices": _fp_slices(),
            "n_pos": n_pos, "n_neg": n_neg,
            "imbalance_ratio": ratio, "scale_pos_weight": ratio,
            "metadata": meta, "split_source": source}


def load_all_proteins(data_dir: Optional[Path] = None) -> Dict[str, Dict]:
    dd = Path(data_dir) if data_dir else DATA_DIR
    names = sorted(f.stem for f in dd.glob("*.h5") if f.stem not in _SKIP)
    if not names:
        raise FileNotFoundError(f"No se encontraron .h5 en: {dd}")

    logger.info(f"Proteínas detectadas: {names}")
    proteins: Dict[str, Dict] = {}
    for name in names:
        try:
            proteins[name] = load_protein(name, dd)
        except Exception as e:
            logger.error(f"Error cargando {name}: {e}")

    by_loc: Dict[str, List] = {}
    for name, d in proteins.items():
        by_loc.setdefault(d["metadata"]["location"], []).append(name)
    logger.info("Distribución por localización:")
    for loc in ["membrane", "cytoplasmic", "nuclear", "unknown"]:
        if loc in by_loc:
            logger.info(f"  {loc:12s}: {by_loc[loc]}")
    return proteins


def get_membrane_proteins(proteins: Dict) -> List[str]:
    return [n for n, d in proteins.items() if d["metadata"]["location"] == "membrane"]
