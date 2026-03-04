"""
virtual_screening.py
====================
Ejecuta el screening masivo sobre la base de PubChem (70,249 moléculas)
usando los modelos entrenados, devolviendo los "Top Hits" candidatos.
"""

import os
import h5py
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import logging
from .config import DATA_DIR, OUTPUT_DIR

logger = logging.getLogger("virtual_screening")

def load_pubchem_library(pubchem_path: Path):
    """Carga la librería de 70k moléculas manteniendo formato sparse."""
    logger.info(f"Cargando librería de PubChem: {pubchem_path.name}")
    
    with h5py.File(pubchem_path, "r") as hf:
        chembl_ids = hf["chembl_id"][()].astype(str)
        
        def read_sparse(key):
            grp = hf[key]
            indptr = grp["indptr"][()]
            return sparse.csr_matrix(
                (grp["data"][()], grp["indices"][()], indptr),
                shape=(len(indptr) - 1, 2039),
            )
            
        ap = read_sparse("ap")
        mg = read_sparse("mg")
        tt = read_sparse("tt")
        
        features_sparse = sparse.hstack([ap, mg, tt], format="csr").astype(np.float32)
        
    logger.info(f"Cargadas {features_sparse.shape[0]} moléculas con {features_sparse.shape[1]} features iniciales.")
    return chembl_ids, features_sparse

def run_virtual_screening(models_dir: Path, output_dir: Path, top_n: int = 100):
    """Ejecuta los modelos contra PubChem y guarda los Top Hits."""
    pubchem_file = DATA_DIR / "pubchem_neg_sample.h5"
    if not pubchem_file.exists():
        logger.error(f"No se encontró {pubchem_file}")
        return

    chembl_ids, X_pubchem = load_pubchem_library(pubchem_file)
    
    # Buscar todos los artefactos de modelo entrenados
    model_files = list(models_dir.glob("*_artifact.pkl"))
    if not model_files:
        logger.warning("No se encontraron modelos entrenados (*_artifact.pkl). Ejecuta el pipeline primero.")
        return

    all_hits = []

    for model_file in model_files:
        with open(model_file, "rb") as f:
            artifact = pickle.load(f)
            
        protein = artifact["protein_name"]
        booster = artifact["booster"]
        keep_mask = artifact["keep_mask"]
        
        logger.info(f"Evaluando contra diana: {protein.upper()}...")
        
        # 1. Aplicar la máscara exacta que usó este modelo en entrenamiento
        if keep_mask is not None:
            X_screen = X_pubchem[:, keep_mask]
        else:
            X_screen = X_pubchem
            
        # 2. Predecir probabilidades
        probs = booster.predict(X_screen)
        
        # 3. Extraer los Top N candidatos (mayor probabilidad)
        top_indices = np.argsort(probs)[-top_n:][::-1]
        
        for rank, idx in enumerate(top_indices, 1):
            all_hits.append({
                "Proteina_Diana": protein,
                "Rank": rank,
                "PubChem_CID": chembl_ids[idx],
                "Probabilidad_Inhibicion": round(probs[idx], 4)
            })

    # Guardar reporte final
    df_hits = pd.DataFrame(all_hits)
    report_path = output_dir / "pubchem_top_hits_screening.csv"
    df_hits.to_csv(report_path, index=False)
    
    logger.info(f"\n🎉 Virtual Screening Completado.")
    logger.info(f"Top candidatos guardados en: {report_path}")
    return df_hits