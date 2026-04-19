# CanInh-Model

**Virtual screening pipeline for kinase inhibitor discovery using LightGBM on sparse molecular fingerprints**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1-ff69b4?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-grey)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

Protein kinases are master regulators of cell proliferation and angiogenesis — their dysregulation drives multiple cancer types. This pipeline predicts the inhibitory activity of small molecules against **8 oncologically critical kinases** using molecular fingerprints, then applies the trained models to perform **virtual screening across 70,249 PubChem compounds** to identify novel drug candidates.

Key results: **AUC-PRC > 0.80** on membrane-located kinases (EGFR, VEGFR2); identification of a **Top 100 high-confidence candidates (probability > 0.95)** converging on Lck as the most druggable target in the screened library.

---

## Scientific context

Kinase inhibitors represent one of the main pillars of modern targeted cancer therapy (imatinib, erlotinib, sorafenib). Their rational design requires identifying molecules capable of binding the ATP pocket and outcompeting the natural substrate — a needle-in-a-haystack problem across chemical space with millions of candidate molecules.

This project addresses two questions:

1. **Structural modeling:** Which molecular substructures (fingerprints) determine binding selectivity for membrane vs. intracellular kinases?
2. **Virtual discovery:** Can a gradient boosting model trained on validated ChEMBL inhibitors generalize to identify novel candidates from an untested PubChem library?

---

## Dataset

**Source:** [Kaggle — Cancer Inhibitors (Protein Kinase)](https://www.kaggle.com/datasets/xiaotawkaggle/inhibitors/data)

Training data derived from ChEMBL (validated inhibitors, IC50 < 10 µM). Virtual screening target: a 70,249-molecule PubChem library of uncharacterized compounds.

| Set | Purpose | Size | Class balance |
|---|---|---|---|
| `[kinase].h5` files | Training / validation per target | 1,000 – 6,000 molecules | Severe imbalance (~1:10 positives) |
| `pubchem_neg...` | Virtual screening candidates | 70,249 molecules | Unknown (searching for false negatives) |

**8 kinase targets covered:** EGFR (erbB1), erbB2, VEGFR2, SRC, ABL, CDK2, Lck, ALK — spanning membrane receptors and intracellular kinases.

---

## Feature engineering

Rather than applying dimensionality reduction (PCA/MCA) that destroys chemical interpretability, this pipeline preserves the full **6,117-dimensional molecular fingerprint space** using `scipy.sparse` matrices, reducing memory usage by **95%** relative to dense representation.

Three complementary RDKit fingerprints are computed per molecule:

| Fingerprint type | Dimensions | Biochemical relevance |
|---|---|---|
| **Atom Pairs (AP)** | 2,039 | Captures topological distances between atom pairs; key for fitting deep binding pockets of intracellular kinases |
| **Morgan circular (MG)** | 2,039 | Encodes the local atomic environment; models lipophilicity and aromatic rings critical for membrane permeability |
| **Topological Torsion (TT)** | 2,039 | Represents implicit 3D conformational flexibility from torsion angles |

---

## Model architecture

**Algorithm:** LightGBM (gradient boosting optimized for sparse binary data)

The primary statistical challenge is severe class imbalance (~1:10). Implemented solutions:

- `scale_pos_weight`: dynamic false-negative penalization computed fold-by-fold
- **Optuna TPE sampler**: Bayesian hyperparameter search, `max_bin=63` adapted for sparse binary inputs
- **Target metric: AUC-PRC** (Precision-Recall AUC) — accuracy is misleading under severe imbalance in drug screening contexts

### Corrections to common public baselines

| Common error in public notebooks | Impact | Solution implemented |
|---|---|---|
| Evaluating on training data | Artificially inflated ~99% accuracy | Stratified split + cross-validation |
| MCA to 800 components | RAM crash (>30 min) + black-box | Native sparse handling (trains in seconds) |
| Ignoring class imbalance | Recall as low as 0.29 | `scale_pos_weight` + F1/AUC-PRC optimization |

---

## Results

### 1. ROC vs. Precision-Recall: why PRC is the right metric for drug screening

ROC curves suggest near-perfect performance across all kinases. PRC curves reveal the true capacity to find inhibitors (signal) without generating excessive false positives (noise) — the metric used in industry.

![ROC vs PRC curves](results/plots/roc_prc_curves.png)

*Left: ROC curves (AUC > 0.95 for most targets). Right: PRC curves showing true performance adjusted for class imbalance.*

### 2. Membrane vs. intracellular kinases: a structural advantage

Post-hoc stratification by cellular localization reveals that **membrane kinases (EGFR, VEGFR2) are significantly more predictable** than intracellular targets (CDK2). This is consistent with the more restricted chemical space of membrane-binding inhibitors.

![Membrane vs intracellular](results/plots/membrane_vs_interior_analysis.png)

*Membrane kinases (red) show systematically higher AUC-PRC and F1-Score than cytoplasmic kinases (blue).*

### 3. Chemical interpretability: what the model learned

Feature importance analysis (Gain) shows that **Morgan fingerprints (MG) dominate predictions for membrane kinases**. Biologically, this validates that the model learned to identify lipophilic and aromatic local groups required for transmembrane receptor interaction — not statistical noise.

![Fingerprint importance by localization](results/plots/fingerprint_importance_by_location.png)

For specific targets like EGFR, the exact fingerprint bits driving inhibition predictions can be mapped to molecular substructures:

![Top features EGFR](results/plots/top_features_egfr_erbB1.png)

### 4. Virtual screening: Top candidates

Applied to 70,249 uncharacterized PubChem molecules, the pipeline distilled the chemical space to a **Top 100 high-confidence candidates (probability > 0.95)**. The highest-scoring compounds converge on **Lck (Lymphocyte-specific protein tyrosine kinase)**, a clinically relevant target in T-cell Acute Lymphoblastic Leukemia (T-ALL).

| PubChem CID | Predicted target | Confidence | Structural rationale |
|---|---|---|---|
| **68058875** | TPK_LCK | **98.5%** | Pyrazolopyrimidine core — privileged scaffold for hinge-region hydrogen bonding in Lck |
| **67593796** | TPK_LCK | **97.2%** | Aromatic substituents complement the deep hydrophobic pocket of the Src kinase family |
| **68058868** | TPK_LCK | **96.8%** | Compound identified in high-throughput screening (HTS) for SFK family — validates model generalization |
| **58289963** | TPK_LCK | **95.1%** | Strict Lipinski Rule-of-5 compliance; robust selectivity profile for in vitro follow-up |

The convergence toward Lck is not coincidental — Lck's role in TCR signaling and its lymphoid-restricted expression make it a high-precision target that minimizes off-target toxicity.

---

## Chemical validation against literature

The nitrogen-containing heterocyclic cores identified by the model are consistent with experimentally validated kinase inhibitor scaffolds:

- **Quinoxaline derivatives** are established privileged scaffolds for Src kinase family inhibition, with demonstrated antitumor activity (Al-Marhabi et al., 2015, *Molecules*)
- **4-Phenoxyquinoline** analogs show that positions 6 and 7 substitution critically determines potency and selectivity for membrane tyrosine kinases (Liu et al., 2014, *Bioorganic & Medicinal Chemistry*)

The model's preference for Morgan fingerprints (local aromatic environments) maps directly to these experimentally critical structural positions.

---

## Limitations and next steps

1. **SMILES extraction:** Map top `PubChem_CID` hits to SMILES via ChEMBL API for structural visualization
2. **Scaffold analysis:** `MurckoScaffold` (RDKit) to cluster candidates by base scaffold and prioritize non-patented structural families
3. **ADMET filtering:** Pass top candidates through ADMET predictors (pkCSM, SwissADME) before considering synthesis
4. **Experimental validation:** IC50 measurements for top 5–10 candidates against recombinant Lck

---

## Repository structure

```
CanInh-Model/
├── notebooks/
│   └── analysis_notebook.ipynb      # EDA, Plotly visualizations, virtual screening
├── src/
│   ├── config.py                    # Paths and hyperparameters (single source of truth)
│   ├── audit_dataset.py             # Memory-efficient HDF5/CSV auditing
│   ├── data_loading.py              # Sparse matrix loading
│   ├── feature_engineering.py       # Fingerprint processing + VarianceThreshold
│   ├── lgbm_model.py                # LightGBM training + Optuna tuning
│   ├── virtual_screening.py         # PubChem prediction engine
│   └── interpretation.py            # Feature importance + cellular localization analysis
├── results/
│   ├── models/                      # Trained .pkl artifacts (not included)
│   └── plots/                       # Performance and validation figures
├── run_pipeline.py                  # Main CLI orchestrator
├── requirements.txt
└── README.md
```

---

## Installation

Optimized for macOS M1/M2 with Miniforge/Conda:

```bash
# 1. Clone and set up environment
git clone https://github.com/CANOLIO/CanInh-Model.git
cd CanInh-Model
conda create -n inhibitors python=3.11
conda activate inhibitors
pip install -r requirements.txt
conda install -c conda-forge rdkit

# 2. Configure data paths
# Edit src/config.py to point to the directory with the .h5 files downloaded from Kaggle.

# 3. Run full training + hyperparameter tuning
python run_pipeline.py --tune

# 4. Train only high-priority membrane targets
python run_pipeline.py --membrane_only
```

For interactive virtual screening, open `notebooks/analysis_notebook.ipynb` after training.

---

## References

- Al-Marhabi, A. R., Abbas, H. S., & Ammar, Y. A. (2015). Synthesis, Characterization and Biological Evaluation of Some Quinoxaline Derivatives. *Molecules, 20*(11), 19805–19822.
- Liu, Z., et al. (2014). Design, synthesis and biological evaluation of novel 6,7-disubstituted-4-phenoxyquinoline derivatives as c-Met kinase inhibitors. *Bioorganic & Medicinal Chemistry, 22*(14), 3642–3653.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.

---

## Author

**Fabián Rojas** — Biochemist & Computational Biologist · Valdivia, Chile

[LinkedIn](https://www.linkedin.com/in/fabianrojasg/) · [GitHub](https://github.com/CANOLIO)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
