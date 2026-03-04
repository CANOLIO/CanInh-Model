"""
run_pipeline.py — Orquestador CLI del pipeline de inhibidores.

Uso:
  python run_pipeline.py                   # todas las proteínas
  python run_pipeline.py --membrane_only   # solo membrana (más rápido)
  python run_pipeline.py --tune            # con Optuna (~30 min M1)
  python run_pipeline.py --data_dir /ruta  # sobreescribir DATA_DIR
"""
import argparse, json, logging, sys, os, pickle

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("pipeline")

from src.config import OUTPUT_DIR
from src import data_loading, feature_engineering, lgbm_model, interpretation


def run(data_dir=None, membrane_only=False, tune=False, chi2=False):
    # Resolver OUTPUT_DIR al CWD actual (run desde el root del proyecto)
    import src.config as cfg
    cfg.OUTPUT_DIR = OUTPUT_DIR.resolve()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.OUTPUT_DIR / "plots").mkdir(exist_ok=True)
    (cfg.OUTPUT_DIR / "models").mkdir(exist_ok=True)

    proteins = data_loading.load_all_proteins(data_dir)

    if membrane_only:
        keep = data_loading.get_membrane_proteins(proteins)
        proteins = {k: v for k, v in proteins.items() if k in keep}
        logger.info(f"membrane_only: {list(proteins.keys())}")

    all_results, all_summaries, boosters = [], {}, {}

    for name, pdata in proteins.items():
        meta = pdata["metadata"]
        logger.info(f"\n{'='*60}\n{name.upper()} — {meta['full_name']}\n"
                    f"  {meta['location']} | {meta['family']} | {meta['uniprot']}")

        split_data = feature_engineering.prepare_split(
            pdata, apply_variance_filter=True, apply_chi2_selection=chi2)

        extra = None
        if tune:
            extra = lgbm_model.tune_hyperparams(split_data)
            if extra:
                p = cfg.OUTPUT_DIR/"models"/f"{name}_params.json"
                with open(p,"w") as f:
                    json.dump({k:v for k,v in extra.items()
                               if isinstance(v,(int,float,str,bool))}, f, indent=2)

        booster, _ = lgbm_model.train(split_data, extra_params=extra)
        model_artifact = {
            "booster": booster,
            "keep_mask": split_data.get("keep_mask", None),  # Extrae la máscara generada
            "protein_name": name,
            "threshold": 0.5 # Umbral por defecto
        }
    
        model_path = cfg.OUTPUT_DIR / "models" / f"{name}_artifact.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_artifact, f)
        boosters[name] = booster

        results = lgbm_model.evaluate(booster, split_data)
        all_results.append(results)

        _, summary = interpretation.fingerprint_importance(booster)
        all_summaries[name] = summary
        interpretation.plot_top_features(booster, name, meta["location"])

    if len(all_results) >= 2:
        interpretation.plot_fingerprint_importance(all_summaries, all_results)
        interpretation.plot_roc_prc_curves(all_results)
        interpretation.analyze_membrane_vs_interior(all_results, all_summaries)

    df = interpretation.build_summary_table(all_results, all_summaries)

    logger.info(f"\n{'='*60}\nRESULTADOS FINALES\n{'='*60}")
    for _, row in df.iterrows():
        e = "🔴" if row["Localizacion"] == "membrane" else "🔵"
        logger.info(f"  {e} {row['Proteina']:15s} AUC-PRC={row['AUC-PRC']:.4f} "
                    f"F1={row['F1']:.4f} | {row['Localizacion']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      default=None)
    p.add_argument("--membrane_only", action="store_true")
    p.add_argument("--tune",          action="store_true")
    p.add_argument("--chi2",          action="store_true")
    args = p.parse_args()
    run(args.data_dir, args.membrane_only, args.tune, args.chi2)
