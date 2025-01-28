# Credit-card-Fraud-detection
Credit Card Fraud Detection using Logistic Regression
This repository contains a Jupyter Notebook (CreditCard.ipynb) that demonstrates the use of Logistic Regression for detecting credit card fraud. The dataset used is highly imbalanced, with a very small percentage of fraudulent transactions. The notebook explores the use of class weighting and stratified cross-validation to handle the imbalance and improve model performance.

Dataset
The dataset used in this project is the creditcard.csv file, which contains credit card transactions made by European cardholders in September 2013. The dataset contains 284,807 transactions, out of which only 492 are fraudulent (0.172% of all transactions). The dataset is highly imbalanced, making it a challenging problem for classification.
Features
Time: Number of seconds elapsed between this transaction and the first transaction in the dataset.

V1-V28: Principal components obtained through PCA (due to confidentiality issues, the original features are not provided).

Amount: Transaction amount.

Class: Target variable (1 for fraud, 0 for non-fraud).

Approach:
1. Handling Imbalanced Data
The dataset is highly imbalanced, with only 0.172% of the transactions being fraudulent. To address this, the following techniques were used:
Handling Imbalanced Data
The dataset is highly imbalanced, with only 0.172% of the transactions being fraudulent. To address this, the following techniques were used:

Class Weighting: The LogisticRegression model was initialized with class_weight='balanced', which automatically adjusts the weights inversely proportional to class frequencies. This helps the model to pay more attention to the minority class (fraudulent transactions).

2. Stratified Cross-Validation
To ensure that the model generalizes well and is not overfitting, Stratified K-Fold Cross-Validation was used. This technique ensures that each fold of the dataset maintains the same proportion of classes as the original dataset.
Model Evaluation
The model was evaluated using the following metrics:

Accuracy: The proportion of correctly classified transactions.

Precision, Recall, and F1-Score: These metrics are particularly important for imbalanced datasets, as they provide a better understanding of the model's performance on the minority class.

Results:
Model Performance:
Accuracy: 97.11%

Precision (Fraud): 0.05

Recall (Fraud): 0.89

F1-Score (Fraud): 0.10
Cross-Validation Results
Mean CV Accuracy: 97.31%

Standard Deviation of CV Scores: 0.0039

Mean F1 Score: 0.1021

Challenges:
Despite using class weighting and stratified cross-validation, the model's performance on the minority class (fraudulent transactions) is still suboptimal. The F1-score for the fraud class is low, indicating that the model struggles to correctly identify fraudulent transactions while maintaining a low false positive rate.

Future Work:
SMOTE (Synthetic Minority Over-sampling Technique): Future work could involve applying SMOTE to generate synthetic samples of the minority class to further balance the dataset.

Alternative Models: Other models such as Random Forest, Gradient Boosting, or Neural Networks could be explored to see if they perform better on this imbalanced dataset.

Hyperparameter Tuning: Further tuning of the Logistic Regression model's hyperparameters could potentially improve performance.

Requirements:

Python 3.9

pandas

numpy

scikit-learn

matplotlib

Acknowledgments:
The dataset used in this project is from Kaggle.

Special thanks to the scikit-learn community for providing excellent documentation and resources.
