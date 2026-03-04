"""
src/lgbm_model.py
=================
Entrenamiento, evaluación y serialización del modelo LightGBM.
"""
import logging, pickle
from typing import Dict, Optional, Tuple
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                              f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.config import (EARLY_STOPPING_ROUNDS, LGBM_BASE_PARAMS,
                         OPTUNA_N_TRIALS, OPTUNA_TIMEOUT_SECS,
                         RANDOM_STATE, VAL_SIZE)

logger = logging.getLogger(__name__)


def _output_dir():
    from src.config import OUTPUT_DIR
    d = OUTPUT_DIR.resolve() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_params(scale_pos_weight: float, extra: Optional[Dict] = None) -> Dict:
    p = {**LGBM_BASE_PARAMS, "scale_pos_weight": scale_pos_weight}
    if extra:
        p.update(extra)
    return p


def train(split_data: Dict,
          early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
          verbose_eval: int = 200,
          extra_params: Optional[Dict] = None) -> Tuple[lgb.Booster, Dict]:
    """
    Entrena con early stopping sobre un split interno de validación.
    El test set nunca se toca aquí.
    """
    X_tr_full = split_data["X_train"]
    y_tr_full = split_data["y_train"]
    feat_names = list(split_data.get("feature_names", []))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full,
        test_size=VAL_SIZE, stratify=y_tr_full, random_state=RANDOM_STATE)

    params = build_params(split_data["scale_pos_weight"], extra_params)

    dtrain = lgb.Dataset(X_tr,  label=y_tr,  feature_name=feat_names or "auto", free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    booster = lgb.train(
        params, dtrain, valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=verbose_eval),
        ])

    logger.info(f"  [{split_data['protein']}] best_iteration={booster.best_iteration}")
    return booster, {"best_iteration": booster.best_iteration}


def evaluate(booster: lgb.Booster, split_data: Dict, threshold: float = 0.5) -> Dict:
    """Evaluación completa sobre el test set oficial."""
    X_te = split_data["X_test"]
    y_te = split_data["y_test"]
    meta = split_data["metadata"]

    y_prob = booster.predict(X_te)
    y_pred = (y_prob >= threshold).astype(int)

    res = {
        "protein":   split_data["protein"],
        "location":  meta["location"],
        "full_name": meta["full_name"],
        "family":    meta.get("family", ""),
        "uniprot":   meta.get("uniprot", ""),
        "relevance": meta.get("relevance", ""),
        "auc_roc":   roc_auc_score(y_te, y_prob),
        "auc_prc":   average_precision_score(y_te, y_prob),
        "f1":        f1_score(y_te, y_pred, zero_division=0),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall":    recall_score(y_te, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_te, y_pred),
        "y_prob": y_prob, "y_pred": y_pred, "y_test": y_te,
        "threshold": threshold, "n_test": len(y_te),
        "n_pos_test": int(y_te.sum()),
    }
    logger.info(f"  [{res['protein']}] AUC-ROC={res['auc_roc']:.4f} "
                f"AUC-PRC={res['auc_prc']:.4f} F1={res['f1']:.4f}")
    return res


def save_model(booster: lgb.Booster, protein_name: str) -> None:
    path = _output_dir() / f"{protein_name}_lgbm.pkl"
    with open(path, "wb") as f:
        pickle.dump(booster, f)
    logger.info(f"  Modelo guardado: {path}")


def load_model(protein_name: str) -> lgb.Booster:
    path = _output_dir() / f"{protein_name}_lgbm.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def tune_hyperparams(split_data: Dict,
                     n_trials: int = OPTUNA_N_TRIALS,
                     timeout: int = OPTUNA_TIMEOUT_SECS) -> Dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna no instalado: pip install optuna")
        return {}

    X_tr = split_data["X_train"]
    y_tr = split_data["y_train"]
    spw  = split_data["scale_pos_weight"]

    def objective(trial):
        p = {"objective": "binary", "metric": "average_precision",
             "device": "cpu", "num_threads": 8, "verbose": -1,
             "random_state": RANDOM_STATE, "scale_pos_weight": spw, "max_bin": 63,
             "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
             "max_depth":         trial.suggest_int("max_depth", 3, 12),
             "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
             "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
             "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
             "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
             "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
             "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
             "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True)}

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tri, vali in skf.split(X_tr, y_tr):
            b = lgb.train(p,
                lgb.Dataset(X_tr[tri], label=y_tr[tri], free_raw_data=False),
                valid_sets=[lgb.Dataset(X_tr[vali], label=y_tr[vali], free_raw_data=False)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(9999)])
            scores.append(average_precision_score(y_tr[vali], b.predict(X_tr[vali])))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = {**study.best_params,
            "objective": "binary", "metric": ["auc","average_precision"],
            "device": "cpu", "num_threads": 8, "verbose": -1,
            "random_state": RANDOM_STATE, "scale_pos_weight": spw, "max_bin": 63}
    logger.info(f"Optuna best AUC-PRC={study.best_value:.4f}")
    return best
