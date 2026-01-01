# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 03:20:07 2026

@author: TAY
Description:
Line graph visualization for TUNED SQL Injection Detection models
"""

import pandas as pd
import matplotlib.pyplot as plt

# =========================================
# Load Tuned Metrics
# =========================================
df = pd.read_csv("tuned_metrics.csv")

print("Tuned Metrics Loaded:")
print(df)

techniques = df["Technique"]

# =========================================
# Helper Function for Line Plot
# =========================================
def plot_metric(metric, title, ylabel):
    plt.figure()
    plt.plot(techniques, df[metric], marker="o")
    
    for i, value in enumerate(df[metric]):
        plt.text(
            i,
            value - 0.05,
            f"{value:.6f}",
            ha="center",
            fontsize=10
        )
    
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xlabel("Detection Techniques")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# =========================================
# Line Graphs – Tuned Models Only
# =========================================
plot_metric("accuracy", "Accuracy Comparison (Tuned Models)", "Accuracy")
plot_metric("precision", "Precision Comparison (Tuned Models)", "Precision")
plot_metric("recall", "Recall Comparison (Tuned Models)", "Recall")

plot_metric("f1", "F1-score Comparison (Tuned Models)", "F1-score")
