# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 00:29:38 2026

NoSQL Injection Detection - ML

@author: TAY
"""

import time
import pymongo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

# =========================================
# MongoDB Connection & Load Dataset
# =========================================
def load_dataset(db_name="nosqlSVM_db", collection_name="nosql_queries"):
    """
    Connects to MongoDB and retrieves the NoSQL dataset.
    If the database is empty, it seeds it using a TXT file in the format:
        <query>::::<label>
    """
    client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    db = client[db_name]
    collection = db[collection_name]

    # Seed database if empty
    if collection.count_documents({}) == 0:
        print("[INFO] MongoDB collection empty. Seeding from TXT file...")
        try:
            dataset = []
            with open("nosqlDataset.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "::::" not in line:
                        print(f"[WARNING] Skipping malformed line: {line}")
                        continue
                    query, label = line.rsplit("::::", 1)
                    dataset.append({"text": query, "label": int(label)})

            if dataset:
                collection.insert_many(dataset)
                print(f"[SUCCESS] Inserted {len(dataset)} records into MongoDB.")
            else:
                raise RuntimeError("No valid records found in TXT file.")

        except Exception as e:
            raise RuntimeError(f"Failed to seed MongoDB: {e}")

    # Load from MongoDB
    cursor = collection.find({}, {"_id": 0, "text": 1, "label": 1})
    df = pd.DataFrame(list(cursor))

    return df

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
        xticklabels=['Benign', 'NoSQLi'],
        yticklabels=['Benign', 'NoSQLi']
    )
    plt.ylabel('Actual', fontsize=13)
    plt.xlabel('Prediction', fontsize=13)
    plt.title(title, fontsize=16, pad=20)
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.show()

# =========================================
# Exploratory Data Analysis (EDA)
# =========================================
df = load_dataset()
df["label"] = df["label"].astype(int)

print("\nDataset Loaded:")
print(df.head())
print(df.info())

counts = df["label"].value_counts().sort_index()
labels = ["Benign", "Malicious"]
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
    counts.tolist(),
    labels=labels,
    autopct=make_autopct(counts.tolist()),
    startangle=90
)

plt.text(0, 0, f"Total\n{total}", ha="center", va="center", fontsize=11)
plt.title("Dataset Distribution of Benign and NoSQL Injection Queries")
plt.axis("equal")
plt.show()

# --- Bar Chart ---
plt.figure()
ax = sns.countplot(x="label", data=df)
plt.xticks([0, 1], ["Benign", "NoSQLi"])
plt.title("Class Distribution of NoSQL Injection Dataset")
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
df["text_length"] = df["text"].apply(len)

plt.figure(figsize=(10,6))
sns.histplot(
    data=df,
    x="text_length",
    hue="label",
    bins=100,
    kde=True
)

plt.legend(["Benign", "NoSQLi"])
plt.title("Query Length Distribution by Class (NoSql)")
plt.xlabel("Query Length (Characters)")
plt.ylabel("Frequency")
plt.show()

# =========================================
# Train-Test Split
# =========================================
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =========================================
# Machine Learning (SVM)
# =========================================
def extract_features(text):
    return [
        len(text),
        text.count("$"),
        text.count("{"),
        text.count("}"),
        text.lower().count("regex"),
        text.lower().count("where"),
        text.lower().count("or"),
        text.lower().count("gt"),
        text.lower().count("sleep("),
        text.lower().count("return true"),
        text.count(";")
    ]

X_ml_train = X_train.apply(extract_features).tolist()
X_ml_test = X_test.apply(extract_features).tolist()

param_grid = {
    "C": [1, 10],        
    "gamma": [0.01, 0.001],
    "kernel": ["rbf"]
    }

grid = GridSearchCV(
    SVC(probability=True, class_weight="balanced"), 
    param_grid,
    refit=True,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=3
)
    
print("\nTraining SVM (NoSQL) with GridSearchCV...")
grid.fit(X_ml_train, y_train)

print("Best SVM Parameters:", grid.best_params_)
best_svm = grid.best_estimator_

start_time = time.time()
y_pred_ml = best_svm.predict(X_ml_test)
end_time = time.time()

latency_ml = (end_time - start_time) / len(X_test)

print("\nMachine Learning (SVM - NoSQL) Results:")
print(classification_report(y_test, y_pred_ml))
print(f"Latency per query: {latency_ml:.6f} seconds")

plot_confusion_matrix(y_test, y_pred_ml, "Confusion Matrix - Machine Learning (SVM - NoSQL)")
metrics_ml = extract_metrics(y_test, y_pred_ml)


# =========================================
# Save NoSQL Metrics
# =========================================
nosql_results = pd.DataFrame([
    {"Technique": "SVM (NoSQL)", **metrics_ml}
])

print("\nFinal Evaluation Metrics:")
print(nosql_results)
