# AWP-Fusion: A Multi-Model Framework for Protein Property Prediction

This repository contains the official implementation for **AWP-Fusion** and three baseline models (AWP, OphPred, Seq2) for predicting protein properties (pHopt, Tm, and Topt). Our method is based on a **balanced weighted pooling** (AWP) mechanism, which is fused with standard mean pooling (Fusion) to achieve robust prediction.

## 📁 Repository Structure

The codebase is organized into four independent model modules:

```text
AWP-Fusion_Project/
├── data/
│   ├── raw/                  # Raw CSV files (train_pH.csv, test_pH.csv, ...)
│   ├── processed/            # 1280-D ESM-2 (650M) features (.pt files)
│   └── weights/              # Best task weights & ESM model weights
├── feature320/               # 320-D ESM-2 (8M) features and weights for control experiments
├── esm_main/                 # ESM model source code (required for feature extraction)
├── model_AWP/                # Model 1: AWP (Balanced Weighted Pooling + KNN)
├── model_AWP-Fusion/         # Model 2: AWP-Fusion (Mean + Weighted Pooling Fusion) 
│                             #   + All ablation, statistical tests, and evaluation scripts
├── model_OphPred/            # Model 3: OphPred (Mean Pooling + KNN Baseline)
├── model_Seq2/               # Model 4: Seq2 (Original Deep Learning Baseline)
├── extract_features_1280.py  # Script to generate 1280-D ESM features (650M model)
├── extract_features_320.py   # Script to generate 320-D ESM features (8M model)
└── results/                  # Output predictions, figures, and statistical results
1. Data Setup
1.1 Raw Data
The raw CSV files are included under data/raw/, with the naming convention train_<task>.csv and test_<task>.csv (where <task> is one of pH, Tm, or Topt).

1.2 ESM Model Weights
We use ESM-2 to generate sequence embeddings. Before generating features, you must download the required model weight files from the official Meta ESM repository and place them in the data/weights/ directory.

For 1280-D features (ESM-2 650M) - Main Experiments:

esm2_t33_650M_UR50D.pt

esm2_t33_650M_UR50D-contact-regression.pt

For 320-D features (ESM-2 8M) - Control Experiment:

esm2_t6_8M_UR50D.pt

esm2_t6_8M_UR50D-contact-regression.pt

Place all downloaded files in the data/weights/ directory as follows:

text
data/weights/
├── esm2_t33_650M_UR50D.pt
├── esm2_t33_650M_UR50D-contact-regression.pt
├── esm2_t6_8M_UR50D.pt
└── esm2_t6_8M_UR50D-contact-regression.pt
⚠️ Note: Both the main weights and the contact-regression weights are required for each model variant. The loading function will fail if the contact-regression file is missing.

1.3 Generate ESM Features
Run the feature extraction scripts from the project root:

Generate 1280-D features (Main experiments):

bash
python extract_features_1280.py data/raw/train_pH.csv   data/raw/test_pH.csv
python extract_features_1280.py data/raw/train_Tm.csv   data/raw/test_Tm.csv
python extract_features_1280.py data/raw/train_Topt.csv data/raw/test_Topt.csv
Features will be saved as .pt files (tensors of shape [sequence_length, 1280]) in the corresponding folders under data/processed/.

Generate 320-D features (Control experiment using ESM-2 8M):

bash
python extract_features_320.py
(Note: This script uses the 8M weights in data/weights/ and outputs the 320-D features to the feature320/ directory).

2. Environment Setup
We recommend using Conda:

bash
conda env create -f environment.yml
conda activate clean-contact
Key dependencies: PyTorch (>=2.0), ESM (fair-esm), scikit-learn, seaborn, statsmodels.

3. Quick Start & Reproduction
Model 1: AWP (Balanced Weighted Pooling + KNN)
bash
cd model_AWP
python run.py --task pH   --best_T 1.0
python run.py --task Tm   --best_T 0.4
python run.py --task Topt --best_T 0.5
Model 2: AWP-Fusion (Mean + Weighted Fusion) & Revisions
Note: This model relies on the trained weights from Model 1 (AWP).

bash
cd model_AWP-Fusion

# Main fusion pipeline
python run.py --task pH   --best_T 1.0
python run.py --task Tm   --best_T 0.4
python run.py --task Topt --best_T 0.5

# --- Scripts added for reviewer revisions ---
# 1. Statistical significance tests (Paired t-tests)
python statistical_tests.py --task pH

# 2. Ablation: MLP depth
python ablation_mlp_depth.py --task pH --T 1.0 --num_layers 1
python ablation_mlp_depth.py --task pH --T 1.0 --num_layers 2
python ablation_mlp_depth.py --task pH --T 1.0 --num_layers 3

# 3. Ablation: Fusion weight
python ablation_fusion_weight.py --task pH --best_T 1.0

# 4. Ablation: Late vs. Intermediate fusion
python intermediate_fusion.py --task pH --best_T 1.0

# 5. Ablation: Regressor choice (KNN vs. Random Forest)
python regressor_comparison.py --task pH --best_T 1.0

# 6. Ablation: Initialization sensitivity
python init_sensitivity.py --task pH --T 1.0 --init_seed 42
python init_sensitivity.py --task pH --T 1.0 --init_seed 123
python init_sensitivity.py --task pH --T 1.0 --init_seed 456
python init_sensitivity.py --task pH --T 1.0 --init_seed 789
python init_sensitivity.py --task pH --T 1.0 --init_seed 1011

# 7. 5-fold cross validation
python cross_validation.py --task pH --T 1.0

# 8. Runtime benchmark
python runtime_benchmark.py --task pH --best_T 1.0
Model 3: OphPred (Mean Pooling + KNN Baseline)
bash
cd model_OphPred
python run.py --task pH
python run.py --task Tm
python run.py --task Topt
Model 4: Seq2 (Deep Learning Baseline)
Note: Requires GPU.

bash
cd model_Seq2
python main.py --task pH   --target_col pHopt
python main.py --task Tm   --target_col tm
python main.py --task Topt --target_col topt
320-Dimensional (ESM-2 8M) Control Experiment
To demonstrate that the effectiveness of the AWP module relies on sufficiently rich PLM embeddings, we conducted a control experiment using the smaller ESM-2 8M variant (320-D features).

Run the baselines using the 320-D features:

bash
cd model_OphPred
python run.py --task pH --feat_suffix "_8M"

cd ../model_AWP
python run.py --task pH --best_T 1.0 --feat_suffix "_8M"
(Note: The results of this 8M control experiment are reported in Supplementary Table S13).

4. Plotting Figures
To generate the training curves and residue weight analysis figures:

bash
cd model_AWP
# Generate all figures for pH task
python plot.py --task pH --best_T 1.0 --type all
# Generate figures for Tm and Topt
python plot.py --task Tm   --best_T 0.4 --type weight
python plot.py --task Topt --best_T 0.5 --type weight
Figures will be saved to results/{task}/figures/.

Citation
If you use our code, please cite:

Wang, W., Shen, L., Wang, F., & Chang, S. (2026). AWP-Fusion: a lightweight adaptive weighted pooling model for enzyme property prediction. Manuscript submitted for publication.

Original Seq2Topt model citation:

Qiu, S., Hu, B., Zhao, J., Xu, W., Yang, A. (2025). Seq2Topt: A Sequence-Based Deep Learning Predictor of Enzyme Optimal Temperature. Briefings in Bioinformatics, 26(2). https://doi.org/10.1093/bib/bbaf114