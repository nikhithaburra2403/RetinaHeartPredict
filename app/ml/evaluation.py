import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, roc_auc_score


def generate_evaluation_artifacts(y_true, y_pred, y_prob, output_dir='app/static/evaluation'):
    """Generate model evaluation metrics and Matplotlib visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)),
    }

    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
    ax_cm.set_title('Confusion Matrix')
    ax_cm.set_xlabel('Predicted label')
    ax_cm.set_ylabel('True label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, cm[i, j], ha='center', va='center', color='black')
    fig_cm.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)

    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, label='ROC curve')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline')
    ax_roc.set_title('ROC Curve')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend()
    fig_roc.tight_layout()
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    fig_roc.savefig(roc_path)
    plt.close(fig_roc)

    fig_metrics, ax_metrics = plt.subplots(figsize=(6, 4))
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
    ax_metrics.bar(labels, values, color=['#2563eb', '#22c55e', '#f59e0b', '#ec4899'])
    ax_metrics.set_ylim(0, 1)
    ax_metrics.set_title('Evaluation Metrics')
    ax_metrics.set_ylabel('Score')
    fig_metrics.tight_layout()
    metrics_path = os.path.join(output_dir, 'metrics.png')
    fig_metrics.savefig(metrics_path)
    plt.close(fig_metrics)

    return metrics, {
        'confusion_matrix': cm_path,
        'roc_curve': roc_path,
        'metrics': metrics_path,
    }
