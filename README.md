# SQL Injection Detection Techniques

## Overview
This repository contains the dataset and experimental results for a study on SQL Injection (SQLi) detection techniques. The study compares signature-based, machine learning, deep learning, and hybrid deep learning approaches under controlled experimental conditions to evaluate their detection performance and limitations.

The following techniques are implemented:
- Signature-Based using **Aho–Corasick**
- Machine Learning using **Support Vector Machine (SVM)**
- Deep Learning using **Bidirectional LSTM (Bi-LSTM)**
- Hybrid Deep Learning using **CNN–BiLSTM**


## Directory Description
- **src/**: Contains the main Python script implementing model training, evaluation, and visualization for all SQL injection detection techniques.
- **data/**: Stores the dataset used in the experiments.
- **results/**: Stores all experimental output and figures generated during evaluation
  - **confusion_matrices/**: A confusion matrix is used to show the performance of a classification model in a table.
  - **classification_reports/**: Classification reports based on the confusion matrix.
  - **metrics_comparison/**: Line graphs comparing evaluation metrics across detection techniques.

## Dataset Details
`SQLiV3.csv`: 
- Contains labelled SQL queries, including benign and malicious queries
- Publicly available on Kaggle: SQL-Injection-Dataset by sajid576 (https://www.kaggle.com/datasets/sajid576/sql-injection-dataset)

## Data Distribution
| Category | Count |
|--------|------|
| Total Queries | 30,917 |
| Benign Queries (Label 0) | 19,536 |
| Malicious Queries (Label 1) | 11,381 |

## Tools and Technologies Used
- **Programming Language**: Python 3.12
- **IDE**: Spyder (Anaconda Distribution)

## Libraries Used
- Pandas
- NumPy
- Scikit-learn
- TensorFlow (Keras)
- pyahocorasick
- Matplotlib
- Seaborn
