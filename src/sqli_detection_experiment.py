# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 13:17:12 2025

@author: TAY
"""

# =========================================
# SQL Injection Detection
# Pattern Matching | ML | DL | Hybrid
# =========================================
import time
import pandas as pd
import ahocorasick
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, BatchNormalization

# =========================================
# Util
# =========================================
def extract_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)   
    return {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
    }

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='',
        cmap="Blues",
        cbar=False,
        xticklabels=['Benign', 'SQLi'],
        yticklabels=['Benign', 'SQLi']
    )
    plt.ylabel('Actual', fontsize=13)
    plt.xlabel('Prediction', fontsize=13)
    plt.title(title, fontsize=16, pad=20)
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.show()

# =========================================
# Load Dataset
# =========================================

# Dateset contains:
# 1. query (SQL input strings - both normal & malicious)
# 2. label (0: benign query, 1: SQL Injection attck)
df = pd.read_csv("SQLiV3.csv")
print("\nDataset Loaded:")
print(df.head())

# =========================================
# Train-Test Split
# =========================================
X = df["query"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =========================================
# Pattern Matching (Aho-Corasick)
# =========================================
signatures = [
    "' or '1'='1",
    "union select",
    "--",
    "/*",
    "drop table",
    "or 1=1",
    "sleep("
]

A = ahocorasick.Automaton()
for idx, sig in enumerate(signatures):
    A.add_word(sig, sig)
A.make_automaton()

def pattern_detect(query):
    query = query.lower()
    for _, _ in A.iter(query):
        return 1
    return 0

start_time = time.time()
y_pred_pm = X_test.apply(pattern_detect)
end_time = time.time()

latency_pm = (end_time - start_time) / len(X_test)

print("\nPattern Matching (Aho-Corasick) Results:")
print(classification_report(y_test, y_pred_pm))
print(f"Latency per query: {latency_pm:.6f} seconds")

plot_confusion_matrix(y_test, y_pred_pm, "Confusion Matrix – Pattern Matching (Aho-Corasick)")
metrics_pm = extract_metrics(y_test, y_pred_pm)

# =========================================
# Machine Learning (SVM)
# =========================================
def extract_features(query):
    return [
        len(query),
        query.count("'"),
        query.count("--"),
        query.lower().count("union"),
        query.lower().count("select"),
        query.lower().count("or")
    ]

X_ml_train = X_train.apply(extract_features).tolist()
X_ml_test = X_test.apply(extract_features).tolist()

svm_model = SVC(kernel="rbf", probability=True)
svm_model.fit(X_ml_train, y_train)

start_time = time.time()
y_pred_ml = svm_model.predict(X_ml_test)
end_time = time.time()

latency_ml = (end_time - start_time) / len(X_test)

print("\nMachine Learning (SVM) Results:")
print(classification_report(y_test, y_pred_ml))
print(f"Latency per query: {latency_ml:.6f} seconds")

plot_confusion_matrix(y_test, y_pred_ml, "Confusion Matrix – Machine Learning (SVM)")
metrics_ml = extract_metrics(y_test, y_pred_ml)

# =========================================
# Tokenization (Shared for DL & Hybrid)
# =========================================
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=200)
X_test_pad = pad_sequences(X_test_seq, maxlen=200)

# =========================================
# Deep Learning (Bi-LSTM)
# =========================================
bilstm_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

bilstm_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("\nTraining Deep Learning Model (Bi-LSTM)...")
bilstm_model.fit(
    X_train_pad,
    y_train,
    epochs=3, # ephoc=5, batch_size=4
    batch_size=16,
    verbose=1
)

start_time = time.time()
y_prob_dl = bilstm_model.predict(X_test_pad)
y_pred_dl = (y_prob_dl > 0.5).astype("int32").ravel()
end_time = time.time()

latency_dl = (end_time - start_time) / len(X_test)

print("\nDeep Learning (Bi-LSTM) Results")
print(classification_report(y_test, y_pred_dl))
print(f"Latency per query: {latency_dl:.6f} seconds")

plot_confusion_matrix(y_test, y_pred_dl, "Confusion Matrix – Deep Learning (Bi-LSTM)")
metrics_dl = extract_metrics(y_test, y_pred_dl)

# =========================================
# Hybrid Approach (CNN-BiLSTM)
# =========================================
cnn_bilstm_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64),
    Conv1D(filters=64, kernel_size=5, activation="relu"),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),
    Bidirectional(LSTM(64)),
    Dropout(0.5),

    Dense(1, activation="sigmoid")
])

cnn_bilstm_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("\nTraining Hybrid Model (CNN-BiLSTM)...")
cnn_bilstm_model.fit(
    X_train_pad,
    y_train,
    epochs=3,
    batch_size=16,
    verbose=1
)

start_time = time.time()
y_prob_hybrid = cnn_bilstm_model.predict(X_test_pad)
y_pred_hybrid = (y_prob_hybrid > 0.5).astype(int).ravel()
end_time = time.time()

latency_hybrid = (end_time - start_time) / len(X_test)

print("\nHybrid (CNN-BiLSTM) Results")
print(classification_report(y_test, y_pred_hybrid))
print(f"Latency per query: {latency_hybrid:.6f} seconds")

plot_confusion_matrix(y_test, y_pred_hybrid, "Confusion Matrix – Hybrid (CNN-BiLSTM)")
metrics_hybrid = extract_metrics(y_test, y_pred_hybrid)

# =========================================
# Line Graphs – Metrics Comparison
# =========================================
techniques = [
    "Aho-Corasick",
    "SVM",
    "Bi-LSTM",
    "CNN-BiLSTM"
]

accuracy = [
    metrics_pm["accuracy"],
    metrics_ml["accuracy"],
    metrics_dl["accuracy"],
    metrics_hybrid["accuracy"]
]

precision = [
    metrics_pm["precision"],
    metrics_ml["precision"],
    metrics_dl["precision"],
    metrics_hybrid["precision"]
]

recall = [
    metrics_pm["recall"],
    metrics_ml["recall"],
    metrics_dl["recall"],
    metrics_hybrid["recall"]
]

f1 = [
    metrics_pm["f1"],
    metrics_ml["f1"],
    metrics_dl["f1"],
    metrics_hybrid["f1"]
]

def plot_metric(values, title, ylabel):
    plt.figure()
    plt.plot(techniques, values, marker="o")
    for i, value in enumerate(values):
        plt.text(
            i,
            value - 0.05,
            f"{value:.6f}",   
            ha="center",
            fontsize=10
        )
    plt.ylim(0, 1)
    plt.grid(True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Detection Technqiues")
    plt.show()

plot_metric(accuracy, "Accuracy Comparison", "Accuracy")
plot_metric(precision, "Precision Comparison", "Precision")
plot_metric(recall, "Recall Comparison", "Recall")
plot_metric(f1, "F1-score Comparison", "F1-score")

