# Breast Cancer Image Classification (Transfer Learning)

A deep learning project using **transfer learning** to classify breast histopathology images as **benign (0)** or **malignant (1)**.


### Dataset

- **Source**: [Kaggle – Breast Cancer Detection](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection?resource=download)
- H&E-stained histopathology images
- Two classes:
  - `0` → Benign
  - `1` → Malignant

### Features
- Transfer learning with pre-trained CNN ( ResNet)
- Logs metrics: **Accuracy, Precision, Recall, F1-score, Specificity, AUC**
- EDA with visualizations in `notebook/eda.ipynb`
- Metrics saved to `results.csv`


### How to Run
- Install dependencies: pip install -r requirements.txt
- Run training: python train.py