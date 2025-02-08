# **Credit Card Fraud Detection**

## **Overview**
This repository implements **credit card fraud detection** using three different models:
1. **Logistic Regression (`CreditCard.ipynb`)**
2. **K-Nearest Neighbors (KNN) (`CreditCard2.ipynb`)**
3. **Random Forest Classifier with Hyperparameter Tuning (`CreditCard_RCF.ipynb`)** 

Each model is designed to classify fraudulent transactions from an **imbalanced dataset**, using techniques like **resampling, SMOTE, stratified cross-validation, and hyperparameter tuning** to enhance performance.

---

## **Dataset**
The dataset used is `creditcard.csv`, consisting of **284,807 transactions**, where **only 492 are fraudulent** (~0.172%). Due to this imbalance, we employ **various resampling techniques** like **downsampling, SMOTE (Synthetic Minority Over-Sampling), and class weighting**.

### **Features:**
- `Time`: Time elapsed since the first transaction.
- `V1-V28`: Principal Component Analysis (PCA) transformed features.
- `Amount`: Transaction amount.
- `Class`: **Target variable** (1 = Fraud, 0 = Non-fraud).

---

## **Approach & Model Comparison**

### **1️⃣ Logistic Regression Model (`CreditCard.ipynb`)**
- Used **class_weight='balanced'** to adjust for class imbalance.
- Applied **Stratified K-Fold Cross-Validation**.
- **Results:**
  - **F1-Score (Fraud):** 0.10
  - **Mean CV Accuracy:** 97.31%

### **2️⃣ K-Nearest Neighbors (KNN) Model (`CreditCard2.ipynb`)**
- **Downsampling**: Limited the majority class to **only one-third** of the minority class.
- **Feature Scaling**: Applied **Min-Max Scaling** for better KNN performance.
- **Stratified Cross-Validation** boosted performance by **~4-5%**.
- **Results:**
  - **F1-Score (Fraud):** **0.91**  
  - **Mean CV F1-Score:** 0.9106  

### **3️⃣ Random Forest Classifier with Hyperparameter Tuning (`CreditCard_RCF.ipynb`)** 🚀 *Best Performing Model!*
- **Step 1:** Took **30% of the dataset** while maintaining fraud/non-fraud ratio using stratified sampling.  
- **Step 2:** **Split into train-test sets before applying SMOTE** to prevent data leakage.  
- **Step 3:** Applied **SMOTE only on the training set** to balance the fraud class.  
- **Step 4:** Used **RandomizedSearchCV with Stratified K-Fold (5 splits)** for hyperparameter tuning.  
- **Step 5:** Evaluated the final model on the **real-world imbalanced test set**.  

#### **Best Hyperparameters Found**:
{'n_estimators': 150, 'max_depth': 15, 'criterion': 'log_loss', 'bootstrap': True}
Results (After Hyperparameter Tuning)
Accuracy: 99.98%
Mean F1-Score (Fraud) from Cross-Validation: 0.9811
F1-Score on Final Test Set: 0.84
Confusion Matrix:

[17056     6]
 [    3    24]]

### **🔍 Model Performance Comparison**
| Model | Class Balancing | Algorithm | Mean F1-Score (Fraud) |
|--------|---------------|-----------|------------------------|
| Logistic Regression | Class Weighting | Logistic Regression | **0.10** |
| KNN | Downsampling & Stratified CV | K-Nearest Neighbors | **0.91** |
| **Random Forest (New)** | SMOTE & RandomizedSearchCV | **Random Forest Classifier** | **0.9811 (CV) / 0.84 (Final Test)** |
---
#### Requirements
Ensure you have Python 3.9+ and the required libraries installed:

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
Running the Notebooks

To run any of the models:

CreditCard.ipynb → Logistic Regression
CreditCard2.ipynb → KNN with Downsampling & Stratified CV
CreditCard_RCF.ipynb → Random Forest with SMOTE & Hyperparameter Tuning


Challenges & Future Work
✅ Despite improvements, fraud detection remains challenging due to class imbalance. Future enhancements include:

Ensemble Learning: Combining models like XGBoost, LightGBM, or Neural Networks.

Feature Engineering: Deriving additional fraud-specific features.

Advanced Sampling Techniques: Testing ADASYN or Tomek Links for better balancing.

Acknowledgments

Dataset sourced from Kaggle. Special thanks to the scikit-learn and imbalanced-learn communities for providing excellent tools for imbalanced classification.


