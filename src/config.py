"""src/config.py — Fuente de verdad única del proyecto."""
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path('/RUTA/AL/DATASET')  # Cambia esto a tu ruta local del dataset
OUTPUT_DIR = Path('./results')   # relativo al root; los módulos lo resuelven en runtime

# ── Fingerprints ──────────────────────────────────────────────────────────────
FINGERPRINT_SIZE  = 2039          # dimensión de cada fp en el .h5
FINGERPRINT_NAMES = ["ap", "mg", "tt"]

# ── Splits y validación ───────────────────────────────────────────────────────
TEST_SIZE    = 0.20   # solo para fallback si los CSVs oficiales no existen
VAL_SIZE     = 0.10   # split interno para early stopping (nunca toca test)
N_CV_FOLDS   = 5
RANDOM_STATE = 42

# ── LightGBM base (M1 8GB + fingerprints binarios) ───────────────────────────
LGBM_BASE_PARAMS = {
    "objective":         "binary",
    "metric":            ["auc", "average_precision"],
    "boosting_type":     "gbdt",
    "device":            "cpu",
    "num_threads":       8,
    "max_bin":           63,
    "num_leaves":        63,
    "max_depth":         -1,
    "learning_rate":     0.05,
    "n_estimators":      1000,
    "min_child_samples": 20,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "verbose":           -1,
    "random_state":      RANDOM_STATE,
}
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL          = 200

# ── Optuna ────────────────────────────────────────────────────────────────────
OPTUNA_N_TRIALS     = 40
OPTUNA_TIMEOUT_SECS = 600

# ── Metadatos bioquímicos ─────────────────────────────────────────────────────
# Clave = nombre exacto del archivo .h5 (sin extensión).
# Localización validada contra UniProt. Relevancia clínica documentada.
#
# membrane    → receptor de membrana, sitio de unión extracelular accesible
#               ~73% de los fármacos aprobados actúan aquí
# cytoplasmic → requiere penetración celular
PROTEIN_METADATA = {
    # RTK (Receptor Tyrosine Kinases) ─ membrana
    "egfr_erbB1": {
        "full_name": "Epidermal growth factor receptor (ErbB1/HER1)",
        "location": "membrane", "family": "RTK", "uniprot": "P00533",
        "relevance": "Oncogén en cáncer de pulmón/mama/colon. Diana de gefitinib, erlotinib.",
    },
    "hgfr": {
        "full_name": "Hepatocyte growth factor receptor (MET)",
        "location": "membrane", "family": "RTK", "uniprot": "P08581",
        "relevance": "Amplificado en cáncer gástrico y pulmón. Diana de crizotinib.",
    },
    "vegfr2": {
        "full_name": "Vascular endothelial growth factor receptor 2 (KDR)",
        "location": "membrane", "family": "RTK", "uniprot": "P35968",
        "relevance": "Clave en angiogénesis tumoral. Diana de sunitinib, sorafenib.",
    },
    # SFK (Src Family Kinases) ─ membrana
    "tpk_lck": {
        "full_name": "Tyrosine-protein kinase Lck",
        "location": "membrane", "family": "SFK", "uniprot": "P06239",
        "relevance": "Kinasa de señalización en linfocitos T. Inmunoterapia.",
    },
    "tpk_src": {
        "full_name": "Proto-oncogene tyrosine-protein kinase Src",
        "location": "membrane", "family": "SFK", "uniprot": "P12931",
        "relevance": "Primera oncoproteína descubierta. Cáncer de colon y mama.",
    },
    # Kinasas citoplasmáticas
    "cdk2": {
        "full_name": "Cyclin-dependent kinase 2",
        "location": "cytoplasmic", "family": "CDK", "uniprot": "P24941",
        "relevance": "Regulador del ciclo celular (fase S). Target oncológico.",
    },
    "gsk3b": {
        "full_name": "Glycogen synthase kinase-3 beta",
        "location": "cytoplasmic", "family": "CMGC", "uniprot": "P49841",
        "relevance": "Alzheimer, diabetes tipo 2, trastorno bipolar.",
    },
    "map_k_p38a": {
        "full_name": "Mitogen-activated protein kinase 14 (p38-alpha/MAPK14)",
        "location": "cytoplasmic", "family": "MAPK", "uniprot": "Q16539",
        "relevance": "Mediador de inflamación. Artritis reumatoide, EPOC.",
    },
}

# ── Visualización ─────────────────────────────────────────────────────────────
LOCATION_COLORS = {
    "membrane":    "#E63946",
    "cytoplasmic": "#457B9D",
    "nuclear":     "#2D6A4F",
    "unknown":     "#ADB5BD",
}
LOCATION_LABELS = {
    "membrane":    "Membrana (alta prioridad)",
    "cytoplasmic": "Citoplasmática",
    "nuclear":     "Nuclear",
    "unknown":     "Desconocida",
}
FP_COLORS = {"ap": "#4ECDC4", "mg": "#FF6B6B", "tt": "#45B7D1"}
