# AWP-Fusion: A Multi-Model Framework for Protein Property Prediction

This repository contains the official implementation for **AWP-Fusion** and three baseline models (AWP, OphPred, Seq2) for predicting protein properties (pHopt, Tm, and Topt). Our method is based on a **balanced weighted pooling** (AWP) mechanism, which is fused with standard mean pooling (Fusion) to achieve robust prediction.

---

## 📁 Repository Structure

The codebase is organized into four independent model modules (one folder per model):

```text
AWP-Fusion_Project/
├── data/                     # Contains raw datasets and pre-computed ESM features
│   ├── raw/                  # Raw CSV files
│   └── processed/            # ESM features (.pt files)
├── model_awp/                # Model 1: AWP (Balanced Weighted Pooling + KNN)
├── model_awp_fusion/         # Model 2: AWP-Fusion (Mean + Weighted Pooling Fusion)
├── model_oph_pred/           # Model 3: OphPred (Mean Pooling + KNN Baseline)
├── model_seq2/               # Model 4: Seq2 (Original Deep Learning Baseline)
└── results/                  # All output predictions and figures

1. Data Download & Setup
We use ESM-2 (esm2_t33_650M_UR50D) to generate 1280-dimensional sequence embeddings. Due to the large size of the ESM features, they are hosted on ScienceDB.

Download the ESM features from ScienceDB: https://doi.org/10.57760/sciencedb.011fm.

Unzip the downloaded files and place them into the data/processed/ directory:
data/processed/
├── pH_sequence_train_feat/
├── pH_sequence_test_feat/
├── Tm_sequence_train_feat/
├── Tm_sequence_test_feat/
├── Topt_sequence_train_feat/
└── Topt_sequence_test_feat/
Place the raw datasets in the data/raw/ directory.

2. Environment Setup
We recommend using Conda to replicate the exact environment.
conda env create -f environment.yml
conda activate your_env_name
Key dependencies:
PyTorch (>=1.13.0)
ESM (fair-esm)
Scikit-learn
Seaborn
Statsmodels

3. Quick Start & Reproduction
All models can be run independently. Navigate to the corresponding model folder and run the commands below.
Model 1: AWP (Balanced Weighted Pooling + KNN)
Note: --best_T is the optimal temperature for the task (e.g., 0.5).
cd model_awp

# Run for pH task
python run.py --task pH --best_T 0.5
# Run for Tm task
python run.py --task Tm --best_T 0.5
# Run for Topt task
python run.py --task Topt --best_T 0.5
Model 2: AWP-Fusion (Mean + Weighted Fusion)
Note: This model relies on the trained weights from Model 1 (AWP).
cd model_awp_fusion

python run.py --task pH --best_T 0.5
python run.py --task Tm --best_T 0.5
python run.py --task Topt --best_T 0.5
Model 3: OphPred (Mean Pooling + KNN Baseline)
cd model_oph_pred

python run.py --task pH
python run.py --task Tm
python run.py --task Topt
Model 4: Seq2 (Deep Learning Baseline)
Note: Requires GPU. Set --target_col to pHopt, tm, or topt accordingly.
cd model_seq2

python main.py --task pH --target_col pHopt
python main.py --task Tm --target_col tm
python main.py --task Topt --target_col topt

4. Plotting Figures
AWP Model Plotting
For pH task, all figures (training curve, weight distribution) will be generated. For Tm and Topt tasks, only the training curve is generated.
cd model_awp

# pH task (Generate all figures)
python plot.py --task pH --best_T 0.5 --type all

# Tm or Topt task (Generate only training curve)
python plot.py --task Tm --best_T 0.5
python plot.py --task Topt --best_T 0.5
All figures will be saved to the results/{task}/figures/ directory.
Datasets & Citations
Topt / Tm data: Obtained from Seq2Topt and DeepTM.

pH opt data: Obtained from EpHod (Zenodo link).

ESM embeddings: facebookresearch/esm

Original Seq2Topt model citation:

Qiu, S., Hu, B., Zhao, J., Xu, W., Yang, A. (2025). Seq2Topt: A Sequence-Based Deep Learning Predictor of Enzyme Optimal Temperature. Briefings in Bioinformatics, 26(2). https://doi.org/10.1093/bib/bbaf114

**Please cite this repository if you use our code:**
> Wang, W., Shen, L., Wang, F., & Chang, S. (2026). AWP-Fusion: a lightweight adaptive weighted pooling model for enzyme property prediction. Manuscript in preparation.