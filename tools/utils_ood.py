import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_ood_metrics(predictions, labels, ood_class):
    # Calculate True Positive Rate (TPR)
    tpr = np.mean((predictions == ood_class) & (labels == ood_class))
    tnr = np.mean((predictions != ood_class) & (labels != ood_class))
    # Calculate False Positive Rate (FPR)
    fpr = np.mean((predictions == ood_class) & (labels != ood_class))
    fnr = np.mean((predictions != ood_class) & (labels == ood_class))
    return tpr, fpr, tnr, fnr


def compare_tp_fp(class_label, list_a, list_b):
    tp = 0
    fp = 0

    for val_a, val_b in zip(list_a, list_b):
        if val_a == class_label and val_b == class_label:
            tp += 1
        elif val_a != class_label and val_b == class_label:
            fp += 1

    return tp, fp


def compute_auroc(predictions, labels):
    # Compute AUROC (Area Under the Receiver Operating Characteristic)
    auroc = roc_auc_score(labels, predictions)

    return auroc


'''
# Generate random predicted probabilities and labels
#np.random.seed(42)
predictions = np.random.randint(0, 2, 100)
labels = np.random.randint(0, 2, 100)

# Compute TPR and FPR
tpr, fpr = compute_tpr_fpr(predictions, labels)

# Compute AUROC
auroc = compute_auroc(predictions, labels)

# Print the results
print("TPR:", tpr)
print("FPR:", fpr)
print("AUROC:", auroc)
'''