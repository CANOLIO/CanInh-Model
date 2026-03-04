"""
src/feature_engineering.py
===========================
Filtrado de features sobre splits ya cargados por data_loading.py.

data_loading entrega X_train/X_test listos (splits oficiales).
Este módulo solo aplica filtros de features ajustados sobre train
y propagados a test — sin modificar el split en sí.
"""
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, chi2
from sklearn.model_selection import StratifiedKFold
from src.config import FINGERPRINT_SIZE, N_CV_FOLDS, RANDOM_STATE

logger = logging.getLogger(__name__)


def make_feature_names(keep_mask: Optional[np.ndarray] = None) -> np.ndarray:
    all_names = np.array(
        [f"ap_{i:04d}" for i in range(FINGERPRINT_SIZE)] +
        [f"mg_{i:04d}" for i in range(FINGERPRINT_SIZE)] +
        [f"tt_{i:04d}" for i in range(FINGERPRINT_SIZE)])
    return all_names[keep_mask] if keep_mask is not None else all_names


def get_fp_ranges(keep_mask: Optional[np.ndarray] = None) -> Dict:
    if keep_mask is None:
        return {"ap": (0, FINGERPRINT_SIZE),
                "mg": (FINGERPRINT_SIZE, 2*FINGERPRINT_SIZE),
                "tt": (2*FINGERPRINT_SIZE, 3*FINGERPRINT_SIZE)}
    orig = {"ap": np.arange(0, FINGERPRINT_SIZE),
            "mg": np.arange(FINGERPRINT_SIZE, 2*FINGERPRINT_SIZE),
            "tt": np.arange(2*FINGERPRINT_SIZE, 3*FINGERPRINT_SIZE)}
    sel = np.where(keep_mask)[0]
    out = {}
    for fp, idx in orig.items():
        surv = np.intersect1d(sel, idx)
        pos  = np.searchsorted(sel, surv)
        out[fp] = (int(pos[0]), int(pos[-1])+1) if len(pos) else (0, 0)
        logger.info(f"  {fp}: {len(idx)} → {len(surv)} features")
    return out


def prepare_split(protein_data: Dict,
                  apply_variance_filter: bool = True,
                  apply_chi2_selection: bool = False,
                  chi2_percentile: int = 60) -> Dict:
    """
    Aplica filtros de features. Los selectores se ajustan SOLO sobre X_train.
    Retorna el mismo dict enriquecido con feature_names, fp_ranges, keep_mask.
    """
    X_tr = protein_data["X_train"]
    X_te = protein_data["X_test"]
    y_tr = protein_data["y_train"]
    name = protein_data["protein"]
    keep_mask = None

    logger.info(f"[{name}] Filtrando features (varianza={apply_variance_filter}, chi2={apply_chi2_selection})...")

    if apply_variance_filter:
        sel = VarianceThreshold(threshold=0.0)
        X_tr = csr_matrix(sel.fit_transform(X_tr))
        X_te = csr_matrix(sel.transform(X_te))
        keep_mask = sel.get_support()
        logger.info(f"  Varianza: {protein_data['X_train'].shape[1]} → {X_tr.shape[1]}")

    if apply_chi2_selection:
        sel2 = SelectPercentile(chi2, percentile=chi2_percentile)
        X_tr = csr_matrix(sel2.fit_transform(X_tr, y_tr))
        X_te = csr_matrix(sel2.transform(X_te))
        inner = sel2.get_support()
        if keep_mask is not None:
            full = np.zeros(protein_data["X_train"].shape[1], dtype=bool)
            full[np.where(keep_mask)[0][inner]] = True
            keep_mask = full
        else:
            keep_mask = inner
        logger.info(f"  Chi2({chi2_percentile}%): → {X_tr.shape[1]}")

    return {**protein_data,
            "X_train": X_tr, "X_test": X_te,
            "feature_names": make_feature_names(keep_mask),
            "fp_ranges": get_fp_ranges(keep_mask),
            "keep_mask": keep_mask}


def make_cv_folds(protein_data: Dict, n_splits: int = N_CV_FOLDS,
                  random_state: int = RANDOM_STATE) -> Tuple:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return protein_data["X_train"], protein_data["y_train"], skf
