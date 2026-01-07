# Injection Detection Techniques (SQL & NoSQL)

## Overview
This repository contains the datasets, source code, and experimental results for a study on SQL Injection (SQLi) and NoSQL Injection (NoSQLi) detection techniques. The study compares signature-based, machine learning, deep learning, and hybrid approaches under controlled experimental conditions to evaluate their detection performance and limitations.

The following techniques are implemented:
- Signature-Based (**Aho–Corasick**): Targeted at SQLi patterns.
- Machine Learning (**SVM**): Implemented for both SQLi and NoSQLi (MongoDB).
- Deep Learning (**Bi-LSTM**): Character-level sequential analysis for SQLi.
- Hybrid (**CNN–BiLSTM**): Advanced feature extraction and sequence dependency for SQLi.


## Directory Description
- **src/**: Contains the main Python scripts implementing model training, evaluation, and visualization for all injection detection techniques.
- **data/**: Stores the datasets used in the experiments.
- **results/**: Stores all experimental output and figures generated during evaluation
  - **confusion_matrices/**: A confusion matrix is used to show the performance of a classification model in a table.
  - **classification_reports/**: Classification reports based on the confusion matrix.
  - **metrics_comparison/**: Line graphs comparing evaluation metrics across detection techniques.
  - **training_model/**: Stores trained model processes.
  - **visualization_dataset/**: Stores plots showing dataset characteristics like class distribution and query length.


## Dataset Details
- `SQLiV3.csv`:
    - Contains labelled SQL queries, including benign and malicious queries.
    - Publicly available on Kaggle: SQL-Injection-Dataset by sajid576 (https://www.kaggle.com/datasets/sajid576/sql-injection-dataset)
- `nosqlDataset.txt`:
    - Combined `mongodb_benign.txt` and `mongodb_injection.txt`.
    - Contains labelled NoSQL queries focusing on MongoDB injection patterns (e.g., `$gt`, `$ne`, `$regex`).
    - Publicly available on GitHub: nosql-injection-detection by anonymous1363101 (https://github.com/anonymous1363101/nosql-injection-detection)

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
- Pymongo
