import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# === CONFIG ===
SUB_FOLDER = 'outputs/submissions/'

def compare_files(file_a, file_b):
    # Construct full paths
    path_a = os.path.join(SUB_FOLDER, file_a)
    path_b = os.path.join(SUB_FOLDER, file_b)

    # Check if files exist
    if not os.path.exists(path_a):
        print(f"Error: File not found: {path_a}")
        return
    if not os.path.exists(path_b):
        print(f"Error: File not found: {path_b}")
        return

    print(f"Comparing:\n   A: {file_a}\n   B: {file_b}")

    # Load - Force "None" to stay as string
    # keep_default_na=False prevents "None" -> NaN conversion
    sub1 = pd.read_csv(path_a, keep_default_na=False, na_values=['']) 
    sub2 = pd.read_csv(path_b, keep_default_na=False, na_values=[''])

    # Sort by RowId to ensure alignment
    sub1 = sub1.sort_values('RowId')
    sub2 = sub2.sort_values('RowId')

    preds1 = sub1['AVERAGE_SPEED_DIFF'].astype(str)
    preds2 = sub2['AVERAGE_SPEED_DIFF'].astype(str)

    # Metrics
    match_count = (preds1 == preds2).sum()
    total_count = len(preds1)
    similarity = (match_count / total_count) * 100

    print(f"\n📊 Report:")
    print(f"   Identical:  {match_count} / {total_count}")
    print(f"   Similarity: {similarity:.2f}%")
    print(f"   Difference: {100 - similarity:.2f}%")

    # Visualization
    if similarity < 100:
        labels = ['None', 'Low', 'Medium', 'High', 'Very_High']
        
        cm = confusion_matrix(preds1, preds2, labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=labels, yticklabels=labels)
        
        plt.xlabel(f"File B: {file_b}")
        plt.ylabel(f"File A: {file_a}")
        plt.title(f"Disagreement Matrix (Sim: {similarity:.2f}%)")
        plt.tight_layout()
        plt.show()
    else:
        print("\nThe files are EXACTLY identical!")

if __name__ == "__main__":
    # Check args
    if len(sys.argv) != 3:
        print("Usage: python compare_submissions.py <file1.csv> <file2.csv>")
        print("Example: python compare_submissions.py sub_xgboost.csv sub_ensemble.csv")
    else:
        compare_files(sys.argv[1], sys.argv[2])
