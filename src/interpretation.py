"""
src/interpretation.py
=====================
Interpretabilidad bioquímica del pipeline LightGBM.

DISEÑO CRÍTICO — rutas:
  Todas las rutas de salida se resuelven en tiempo de EJECUCIÓN, nunca
  al importar el módulo. Esto evita que OUTPUT_DIR relativo quede fijo
  al directorio de trabajo del momento de importación (que desde notebooks/
  sería notebooks/results/ en lugar de CanInh-Model/results/).

  La función pública plots_dir() retorna siempre la ruta absoluta correcta.
  Ninguna función privada (_) es llamada desde el notebook.
"""
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional

from src.config import FP_COLORS, LOCATION_COLORS, LOCATION_LABELS

logger = logging.getLogger(__name__)

# Orden de prioridad para ordenar proteínas en gráficos
_LOC_ORDER = {"membrane": 0, "cytoplasmic": 1, "nuclear": 2, "unknown": 3}


# ── Ruta de salida (pública, evaluada en runtime) ─────────────────────────────

def plots_dir() -> "Path":
    """Retorna Path absoluto a results/plots/, creándolo si no existe."""
    from src.config import OUTPUT_DIR
    d = OUTPUT_DIR.resolve() / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(name: str, dpi: int = 150) -> str:
    path = str(plots_dir() / name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Guardado: {path}")
    return path


# ── 1. Importancia por tipo de fingerprint ────────────────────────────────────

def fingerprint_importance(booster: lgb.Booster,
                           importance_type: str = "gain"):
    """
    Calcula la importancia total de cada tipo de fingerprint (ap / mg / tt).

    Responde: ¿qué fingerprint aporta más información para esta proteína?
      ap  (atom pairs)          → distancias entre pares de átomos
      mg  (Morgan/circular)     → entorno local de cada átomo
      tt  (topological torsion) → ángulos de torsión implícitos

    Retorna (df_features, df_summary).
    """
    importances  = booster.feature_importance(importance_type=importance_type)
    names        = np.array(booster.feature_name())
    df = pd.DataFrame({"feature": names, "importance": importances})
    df["fp_type"] = df["feature"].str.split("_").str[0]

    summary = (df.groupby("fp_type")["importance"]
               .agg(total="sum", mean="mean", count="count")
               .assign(pct=lambda x: x["total"] / x["total"].sum() * 100)
               .sort_values("total", ascending=False))
    return df, summary


# ── 2. Gráfico comparativo todas las proteínas ────────────────────────────────

def plot_fingerprint_importance(summaries: Dict, all_results: List[Dict]) -> str:
    """
    Panel izquierdo: stacked bar de importancia AP/MG/TT por proteína.
    Panel derecho: AUC-PRC horizontal, coloreado por localización.
    Proteínas ordenadas: membrana → citoplásmica → nuclear.
    """
    sorted_r = sorted(all_results, key=lambda r: (_LOC_ORDER.get(r["location"],9), -r["auc_prc"]))
    pnames   = [r["protein"]  for r in sorted_r]
    locs     = [r["location"] for r in sorted_r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Importancia por Tipo de Fingerprint\nagrupada por Localización Celular",
                 fontsize=14, fontweight="bold")

    x      = np.arange(len(pnames))
    bottom = np.zeros(len(pnames))
    for fp in ["ap", "mg", "tt"]:
        vals = []
        for p in pnames:
            s = summaries.get(p)
            vals.append(s.loc[fp, "pct"] if s is not None and fp in s.index else 0.0)
        ax1.bar(x, vals, 0.6, bottom=bottom, label=fp.upper(),
                color=FP_COLORS[fp], alpha=0.85)
        bottom += np.array(vals)

    ax1.set_xticks(x)
    ax1.set_xticklabels(pnames, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("% Importancia Total (gain)")
    ax1.set_title("Contribución por Tipo de Fingerprint")
    for i, loc in enumerate(locs):
        ax1.axvspan(i-.4, i+.4, alpha=0.07,
                    color=LOCATION_COLORS.get(loc,"gray"), zorder=0)

    loc_patches = [mpatches.Patch(color=LOCATION_COLORS[l], label=LOCATION_LABELS[l], alpha=0.6)
                   for l in ["membrane","cytoplasmic","nuclear"] if any(r["location"]==l for r in all_results)]
    fp_patches  = [mpatches.Patch(color=FP_COLORS[fp], label=f"{fp.upper()} fingerprint")
                   for fp in ["ap","mg","tt"]]
    ax1.legend(handles=loc_patches+fp_patches, fontsize=7, loc="upper right")

    auc_prcs   = [r["auc_prc"] for r in sorted_r]
    bar_colors = [LOCATION_COLORS.get(l,"#ADB5BD") for l in locs]
    bars = ax2.barh(pnames, auc_prcs, color=bar_colors, alpha=0.85)
    ax2.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("AUC-PRC (Precision-Recall)")
    ax2.set_title("Rendimiento por Proteína")
    ax2.set_xlim(0, 1.05)
    for bar, val in zip(bars, auc_prcs):
        ax2.text(val+0.005, bar.get_y()+bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=8)
    ax2.legend(handles=loc_patches, fontsize=8, loc="lower right")

    plt.tight_layout()
    return _save("fingerprint_importance_by_location.png")


# ── 3. Membrana vs interior celular ───────────────────────────────────────────

def analyze_membrane_vs_interior(all_results: List[Dict],
                                 summaries: Dict) -> pd.DataFrame:
    """
    Boxplots comparativos: proteínas de membrana vs interior celular.
    Hipótesis: membrana → sitios más accesibles → patrones más nítidos.
    """
    rows = []
    for r in all_results:
        p   = r["protein"]
        loc = r["location"]
        row = {"protein": p, "location": loc,
               "grupo": "Membrana" if loc == "membrane" else "Interior Celular",
               "auc_roc": r["auc_roc"], "auc_prc": r["auc_prc"], "f1": r["f1"]}
        if p in summaries:
            s = summaries[p]
            for fp in ["ap","mg","tt"]:
                row[f"{fp}_pct"] = s.loc[fp,"pct"] if fp in s.index else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    grupos = df["grupo"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Membrana vs Interior Celular: Rendimiento y Features",
                 fontsize=13, fontweight="bold")

    metrics = [("auc_prc", "AUC-PRC"), ("f1", "F1 Score"), ("mg_pct", "% Importancia Morgan (MG)")]
    grp_obj = df.groupby("grupo")
    colors  = [LOCATION_COLORS["membrane"], LOCATION_COLORS["cytoplasmic"]]
    order   = ["Membrana", "Interior Celular"]

    for ax, (col, title) in zip(axes, metrics):
        data   = [grp_obj.get_group(g)[col].values for g in order if g in grupos]
        labels = [g for g in order if g in grupos]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        rng = np.random.default_rng(42)
        for i, (d, lbl) in enumerate(zip(data, labels)):
            ax.scatter([i+1+j for j in rng.uniform(-0.1, 0.1, len(d))],
                       d, alpha=0.9, s=50, zorder=5, color=colors[i])
            for j, (xj, yj) in enumerate(zip([i+1]*len(d), d)):
                ax.annotate(rows[[r["protein"] for r in rows].index(
                    df[df["grupo"]==lbl].iloc[j]["protein"])]["protein"],
                    (xj+rng.uniform(-0.05,0.05), yj), fontsize=6, alpha=0.7)
        ax.set_title(title)

    plt.tight_layout()
    _save("membrane_vs_interior_analysis.png")

    # Log de interpretación bioquímica
    if "Membrana" in grupos and "Interior Celular" in grupos:
        m_prc = df[df["grupo"]=="Membrana"]["auc_prc"].mean()
        i_prc = df[df["grupo"]=="Interior Celular"]["auc_prc"].mean()
        m_mg  = df[df["grupo"]=="Membrana"].get("mg_pct", pd.Series([0])).mean()
        i_mg  = df[df["grupo"]=="Interior Celular"].get("mg_pct", pd.Series([0])).mean()
        logger.info(f"\n INTERPRETACIÓN BIOQUÍMICA:")
        logger.info(f"  AUC-PRC membrana={m_prc:.3f}  interior={i_prc:.3f}  diff={m_prc-i_prc:+.3f}")
        logger.info(f"  MG% membrana={m_mg:.1f}  interior={i_mg:.1f}")
        if m_prc > i_prc:
            logger.info("  → Proteínas de membrana más predecibles (sitio más accesible)")
        if m_mg > i_mg:
            logger.info(f"  → Mayor importancia Morgan en membrana (entorno local más informativo)")
    return df


# ── 4. Top features por proteína ──────────────────────────────────────────────

def plot_top_features(booster: lgb.Booster, protein_name: str,
                      location: str, top_n: int = 25) -> str:
    importances = booster.feature_importance(importance_type="gain")
    names       = np.array(booster.feature_name())
    top_idx     = np.argsort(importances)[-top_n:][::-1]
    top_names   = names[top_idx]
    top_vals    = importances[top_idx]
    colors      = [FP_COLORS.get(n.split("_")[0], "#ADB5BD") for n in top_names]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.arange(len(top_names)), top_vals, color=colors, alpha=0.85)
    ax.set_yticks(np.arange(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia (gain)")
    ax.set_title(f"Top {top_n} Features — {protein_name.upper()}\n"
                 f"Localización: {LOCATION_LABELS.get(location, location)}",
                 fontweight="bold")
    patches = [mpatches.Patch(color=FP_COLORS[fp], label=f"{fp.upper()} fingerprint")
               for fp in ["ap","mg","tt"]]
    ax.legend(handles=patches, loc="lower right")
    plt.tight_layout()
    return _save(f"top_features_{protein_name}.png")


# ── 5. Curvas ROC / PRC ───────────────────────────────────────────────────────

def plot_roc_prc_curves(all_results: List[Dict]) -> str:
    from sklearn.metrics import roc_curve, precision_recall_curve

    sorted_r = sorted(all_results, key=lambda r: _LOC_ORDER.get(r["location"],9))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for r in sorted_r:
        loc   = r["location"]
        color = LOCATION_COLORS.get(loc, "#ADB5BD")
        lw    = 2.5 if loc == "membrane" else 1.2
        alpha = 0.9 if loc == "membrane" else 0.6

        fpr, tpr, _  = roc_curve(r["y_test"], r["y_prob"])
        prec, rec, _ = precision_recall_curve(r["y_test"], r["y_prob"])

        ax1.plot(fpr, tpr, color=color, lw=lw, alpha=alpha,
                 label=f"{r['protein']} ({r['auc_roc']:.3f})")
        ax2.plot(rec, prec, color=color, lw=lw, alpha=alpha,
                 label=f"{r['protein']} ({r['auc_prc']:.3f})")

    ax1.plot([0,1],[0,1],"k--",lw=0.8)
    ax1.set(xlabel="FPR", ylabel="TPR", title="Curvas ROC")
    ax1.legend(fontsize=7, loc="lower right")
    ax2.set(xlabel="Recall", ylabel="Precision",
            title="Curvas Precision-Recall\n(recomendadas con desbalance)")
    ax2.legend(fontsize=7, loc="upper right")

    loc_patches = [mpatches.Patch(color=LOCATION_COLORS[l], label=LOCATION_LABELS[l])
                   for l in ["membrane","cytoplasmic"] if any(r["location"]==l for r in all_results)]
    fig.legend(handles=loc_patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5,-0.02))
    plt.tight_layout()
    return _save("roc_prc_curves.png")


# ── 6. Tabla resumen final ────────────────────────────────────────────────────

def build_summary_table(all_results: List[Dict], summaries: Dict) -> pd.DataFrame:
    """
    Tabla ordenada: membrana primero, luego por AUC-PRC descendente.
    Columna 'location' en inglés para consistencia interna;
    'Localizacion' en español para display en el notebook.
    """
    from src.config import OUTPUT_DIR
    rows = []
    for r in all_results:
        p   = r["protein"]
        row = {
            "Proteina":     p,
            "Nombre":       r["full_name"],
            "Localizacion": r["location"],   # ← sin tilde, consistente con config
            "Familia":      r.get("family",""),
            "AUC-ROC":  round(r["auc_roc"],  4),
            "AUC-PRC":  round(r["auc_prc"],  4),
            "F1":       round(r["f1"],        4),
            "Precision":round(r["precision"], 4),
            "Recall":   round(r["recall"],    4),
            "_order":   _LOC_ORDER.get(r["location"],9),
        }
        if p in summaries:
            s = summaries[p]
            for fp in ["ap","mg","tt"]:
                row[f"{fp.upper()} %"] = round(s.loc[fp,"pct"],1) if fp in s.index else 0.0
        rows.append(row)

    df = (pd.DataFrame(rows)
          .sort_values(["_order","AUC-PRC"], ascending=[True,False])
          .drop(columns=["_order"])
          .reset_index(drop=True))

    csv_path = OUTPUT_DIR.resolve() / "final_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nTabla guardada: {csv_path}")
    return df
