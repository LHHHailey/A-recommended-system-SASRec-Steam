# SASRec for Steam Recommendation
A PyTorch implementation of **SASRec (Sequential Recommendation)** on the Steam dataset.

---

## 📝 Notebook Execution Order
Follow this cell order to run the project correctly:

| Cell # | Module          | Description                                                                 |
|--------|-----------------|-----------------------------------------------------------------------------|
| 1      | DataProcessing  | Preprocess raw data (skip if `data/Steam.txt` already exists)               |
| 2      | Environment Init| **Must run first** — Initialize dependencies, device, and global settings    |
| 3      | Sampler Module  | Threading-based sampler to fix Jupyter kernel crash issues                  |
| 4      | Modules Module  | Core Transformer components (multi-head attention, point-wise feed-forward) |
| 5      | Model Module    | SASRec model computation graph definition                                  |
| 6      | Util Module     | Data loading utilities & evaluation metrics (NDCG, HR)                      |
| 7      | Hyperparameters | Default config: `--maxlen=50 --dropout_rate=0.2` (tunable)                 |
| 8      | Load Data       | Load processed dataset & show stats (user count, item count, etc.)          |
| 9      | Build Model     | Re-runnable — will automatically reset the computation graph               |
| 10     | Training Loop   | Main training with progress bar; auto-evaluate & save model every 5 epochs  |
| 11     | Visualization   | Plot NDCG/HR curves after training completion                              |

---

## 🚀 Quick Start
1. **Prepare Data**
   - Place raw Steam data in `data/` directory
   - Run Cell 1 to generate `data/Steam.txt` (skip if file exists)

2. **Run Notebook**
   - Execute Cell 2 first to set up the environment
   - Follow the cell order listed above to complete training
