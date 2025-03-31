import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
This script computes the AUC scores for ROC and PR curves
based on true labels and predicted probabilities.
"""

EPSILON = 1e-6


def statistics(y_true, y_pred):
    """
    Computes precision, recall, and false positive rate (FPR)
    based on true and predicted labels.

    Args:
        y_true (list[int]): True binary labels.
        y_pred (list[int]): Predicted binary labels.

    Returns:
        tuple: Precision, recall, and FPR values.
    """
    # Initialize counts for true positives, false positives, true negatives, and false negatives
    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        else:
            tn += 1

    # Compute precision, recall, and false positive rate
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    fpr = fp / (fp + tn + EPSILON)
    return precision, recall, recall, fpr


def auc(y_true, y_prob):
    """
    Computes the AUC scores for ROC and PR curves.

    Args:
        y_true (list[int]): True binary labels.
        y_prob (list[float]): Predicted probabilities.

    Returns:
        tuple: ROC and PR curve data points.
    """
    # Generate thresholds and compute ROC and PR points
    thresholds = np.linspace(0, 1, 100)
    roc = []
    pr = []

    for i in range(len(thresholds)):
        y_pred = np.zeros(len(y_true))
        y_pred[y_prob >= thresholds[i]] = 1
        precision, recall, tpr, fpr = statistics(y_true, y_pred)
        roc.append([fpr, tpr])
        pr.append([recall, precision])

    # Sort ROC and PR points by their respective x-axis values
    roc.sort(key=lambda x: x[0])
    pr.sort(key=lambda x: x[0])

    # Compute area under the curves
    auc_pr, auc_roc = 0, 0
    for i in range(1, len(roc)):
        auc_roc = auc_roc + (roc[i][0] - roc[i - 1][0]) * (roc[i][1] + roc[i][1]) * 0.5
        auc_pr = auc_pr + (pr[i][0] - pr[i - 1][0]) * (pr[i][1] + pr[i][1]) * 0.5

    print(f"AUC ROC {auc_roc} {auc_pr}")
    return np.array(roc), np.array(pr)


if __name__ == "__main__":
    """
    Driver code to test AUC computation and plot ROC and PR curves.
    """
    # Generate a larger dataset for testing
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_prob = (
        y_true * 0.05 + np.random.rand(100) * 0.1
    )  # Adding some noise to true labels

    # Compute ROC and PR curve data points
    roc, pr = auc(y_true, y_prob)

    # Plot ROC and PR curves in a subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot ROC curve
    sns.lineplot(x=roc[:, 0], y=roc[:, 1], marker="o", ax=ax1, label="ROC Curve")
    ax1.fill_between(roc[:, 0], roc[:, 1], alpha=0.2)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True)

    # Plot PR curve
    sns.lineplot(x=pr[:, 0], y=pr[:, 1], marker="o", ax=ax2, label="PR Curve")
    ax2.fill_between(pr[:, 0], pr[:, 1], alpha=0.2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("PR Curve")
    ax2.legend()
    ax2.grid(True)

    # Save the plot as a PNG file and display it
    plt.tight_layout()
    plt.savefig("auc_plots.png")
    plt.show()
