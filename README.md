# Credit Card Fraud Detection

## Overview
This repository contains implementations of **credit card fraud detection** using two different models:
1. **Logistic Regression (`CreditCard.ipynb`)**
2. **K-Nearest Neighbors (KNN) (`CreditCard2.ipynb`)**

Both models aim to classify fraudulent transactions from an **imbalanced dataset**, using techniques like **class weighting, downsampling, and stratified cross-validation** to improve performance.

---
## Dataset
The dataset used is `creditcard.csv`, which consists of **284,807 transactions**, out of which **only 492 are fraudulent** (~0.172% of all transactions). Since this is a highly imbalanced dataset, we employ **various resampling and balancing techniques**.

### **Features:**
- `Time`: Time elapsed since the first transaction.
- `V1-V28`: Principal Component Analysis (PCA) transformed features.
- `Amount`: Transaction amount.
- `Class`: **Target variable** (1 = Fraud, 0 = Non-fraud).

---
## **Approach & Improvements**

### **1️⃣ Logistic Regression Model (`CreditCard.ipynb`)**
- Used **class_weight='balanced'** to automatically adjust weights inversely proportional to class frequency.
- Applied **Stratified K-Fold Cross-Validation**.
- **Results:**
  - **F1-Score (Fraud):** 0.10
  - **Mean CV Accuracy:** 97.31%

### **2️⃣ K-Nearest Neighbors (KNN) Model (`CreditCard2.ipynb`)**
- **Downsampling:** Limited the majority class to **only one-third** of the minority class.
- **Feature Scaling:** Applied **Min-Max Scaling** to improve KNN performance.
- **Stratified Cross-Validation:** Helped boost model performance by ~4-5%.
- **Results:**
  - **F1-Score (Fraud):** **0.91** (Significant improvement from previous model)
  - **Mean CV F1-Score:** 0.9106

### **🔍 Key Improvements in `CreditCard2.ipynb`**
| Model | Class Balancing | Algorithm | Mean F1-Score (Fraud) |
|--------|---------------|-----------|------------------------|
| Logistic Regression | Class Weighting | Logistic Regression | **0.10** |
| KNN (New) | Downsampling & Stratified CV | K-Nearest Neighbors | **0.91** (Huge improvement) |

---
## **How to Run the Notebooks**
### **Requirements**
Make sure you have Python 3.9+ and the required libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### **Running the Notebooks**
1. Download the dataset (`creditcard.csv`) and place it in the project folder.
2. Open Jupyter Notebook:
```bash
jupyter notebook
```
3. Run either of the two models:
   - `CreditCard.ipynb` (Logistic Regression)
   - `CreditCard2.ipynb` (KNN with Downsampling & Stratified CV)

---
## **Challenges & Future Work**
✅ Even with KNN, detecting fraud is difficult due to class imbalance. Future improvements include:
- **SMOTE (Synthetic Minority Over-Sampling Technique)** to generate synthetic fraud samples.
- **Exploring alternative models** like Random Forest, XGBoost, or Neural Networks.
- **Hyperparameter tuning** for KNN and Logistic Regression to improve precision.

---
## **Acknowledgments**
The dataset is sourced from Kaggle. Special thanks to the **scikit-learn community** for their excellent documentation and resources.

---

**🚀 If you find this project useful, feel free to ⭐ this repository!**

