# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 16:14:24 2025

@author: TAY
"""

# =========================================
# SQL Injection Detection
# Signature-Based | ML | DL | Hybrid
# =========================================
import time
import pandas as pd
import ahocorasick
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
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
# Exploratory Data Analysis (EDA)
# =========================================
print("\nDataset Info:")
print(df.info())

counts = df["label"].value_counts().sort_index()
labels = ["Benign", "SQL Injection"]
total = counts.sum()

print("\nLabel Distribution:")
print(counts)

# --- Pie Chart ---
plt.figure(figsize=(6, 6))

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)
    return my_autopct
    
plt.pie(
    counts,
    labels=labels,
    autopct=make_autopct(counts),
    startangle=90
)

plt.text(0, 0, f"Total\n{total}", ha="center", va="center", fontsize=11)
plt.title("Dataset Distribution of Benign and SQL Injection Queries")
plt.axis("equal")
plt.show()

# --- Class Distribution ---
plt.figure()
ax = sns.countplot(x="label", data=df)
plt.xticks([0, 1], ["Benign", "SQLi"])
plt.title("Class Distribution of SQL Injection Dataset")
plt.xlabel("Query Type")
plt.ylabel("Number of Queries")

for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=10
    )
plt.show()

# --- Query Length Distribution ---
df["query_length"] = df["query"].apply(len)

plt.figure(figsize=(10,6))
sns.histplot(
    data=df,
    x="query_length",
    hue="label",
    bins=100,
    kde=True
)
plt.xlim(0,500)
plt.legend(["Benign", "SQLi"])
plt.title("Query Length Distribution by Class")
plt.xlabel("Query Length (Characters)")
plt.ylabel("Frequency")
plt.show()

# --- SQL Keyword Frequency ---
keywords = ["select", "union", "or", "and", "drop"]
for kw in keywords:
    df[f"{kw}_count"] = df["query"].str.lower().str.count(kw)

keyword_df = df.groupby("label")[[f"{k}_count" for k in keywords]].mean()
keyword_df.T.plot(kind="bar")
plt.xticks(rotation=45)
plt.title("SQL Keyword Frequency by Class")
plt.xlabel("SQL Keywords")
plt.ylabel("Average Occurrence")
plt.legend(["Benign", "SQLi"])
plt.show()

# =========================================
# Train-Test Split
# =========================================
X = df["query"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =========================================
# Signature-Based (Aho-Corasick)
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

print("\nSignature-Based (Aho-Corasick) Results:")
print(classification_report(y_test, y_pred_pm))
print(f"Latency per query: {latency_pm:.6f} seconds")

plot_confusion_matrix(y_test, y_pred_pm, "Confusion Matrix - Signature-Based (Aho-Corasick)")
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

param_grid = {
    "C": [1, 10],        
    "gamma": [0.01, 0.001],
    "kernel": ["rbf"]
    }

grid = GridSearchCV(
    SVC(probability=True), 
    param_grid,
    refit=True,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=3
)
    
print("\nTraining SVM with GridSearchCV...")
grid.fit(X_ml_train, y_train)

print("Best SVM Parameters:", grid.best_params_)
best_svm = grid.best_estimator_

start_time = time.time()
y_pred_ml = best_svm.predict(X_ml_test)
end_time = time.time()

latency_ml = (end_time - start_time) / len(X_test)

print("\nMachine Learning (SVM) Results - Tuned")
print(classification_report(y_test, y_pred_ml))
print(f"Latency per query: {latency_ml:.6f} seconds")

plot_confusion_matrix(y_test, y_pred_ml, "Confusion Matrix - Machine Learning (SVM - Tuned)")
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
    Embedding(
        input_dim=len(tokenizer.word_index) + 1, 
        output_dim=128, 
    ),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

bilstm_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("\nTraining Deep Learning Model (Bi-LSTM - Tuned)...")
bilstm_model.fit(
    X_train_pad,
    y_train,
    epochs=5, 
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

start_time = time.time()
y_prob_dl = bilstm_model.predict(X_test_pad)
y_pred_dl = (y_prob_dl > 0.5).astype("int32").ravel()
end_time = time.time()

latency_dl = (end_time - start_time) / len(X_test)

print("\nDeep Learning (Bi-LSTM) Results - Tuned")
print(classification_report(y_test, y_pred_dl))
print(f"Latency per query: {latency_dl:.6f} seconds")

plot_confusion_matrix(y_test, y_pred_dl, "Confusion Matrix - Deep Learning (Bi-LSTM - Tuned)")
metrics_dl = extract_metrics(y_test, y_pred_dl)

# =========================================
# Hybrid Approach (CNN-BiLSTM)
# =========================================
cnn_bilstm_model = Sequential([
    Embedding(
        input_dim=len(tokenizer.word_index) + 1, 
        output_dim=128
    ),
    Conv1D(
        filters=64, 
        kernel_size=5, 
        activation="relu"
    ),
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

print("\nTraining Hybrid Model (CNN-BiLSTM - Tuned)...")
cnn_bilstm_model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

start_time = time.time()
y_prob_hybrid = cnn_bilstm_model.predict(X_test_pad)
y_pred_hybrid = (y_prob_hybrid > 0.5).astype(int).ravel()
end_time = time.time()

latency_hybrid = (end_time - start_time) / len(X_test)

print("\nHybrid (CNN-BiLSTM) Results - Tuned")
print(classification_report(y_test, y_pred_hybrid))
print(f"Latency per query: {latency_hybrid:.6f} seconds")

plot_confusion_matrix(y_test, y_pred_hybrid, "Confusion Matrix - Hybrid (CNN-BiLSTM - Tuned)")
metrics_hybrid = extract_metrics(y_test, y_pred_hybrid)

# =========================================
# Save Tuned Metrics
# =========================================
tuned_results = pd.DataFrame([
    {"Technique": "Aho-Corasick", **metrics_pm},
    {"Technique": "SVM", **metrics_ml},
    {"Technique": "Bi-LSTM", **metrics_dl},
    {"Technique": "CNN-BiLSTM", **metrics_hybrid}
])

tuned_results.to_csv("tuned_metrics.csv", index=False)
print("Tuned metrics saved to tuned_metrics.csv")

