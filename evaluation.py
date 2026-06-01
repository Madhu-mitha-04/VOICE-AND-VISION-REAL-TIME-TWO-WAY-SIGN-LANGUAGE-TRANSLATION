import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('sign_language_model.h5')

# Load test data
test_data = np.load('processed_data/test/features.npz')
X_test = test_data['features']
y_test = test_data['labels']

# Load class mapping
class_mapping = np.load('class_mapping.npy', allow_pickle=True).item()
idx_to_class = class_mapping['idx_to_class']

# Predict on test set
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class indices

# Convert y_test from one-hot encoded to class indices (if needed)
if y_test.ndim > 1:
    y_test = np.argmax(y_test, axis=1)

## ======================
## 1. Accuracy Metrics
## ======================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n=== Model Performance Metrics ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

## ======================
## 2. Classification Report
## ======================
print("\n=== Classification Report ===")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=[idx_to_class[i] for i in range(len(idx_to_class))]
))

## ======================
## 3. Confusion Matrix
## ======================
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

plot_confusion_matrix(
    y_test, 
    y_pred, 
    classes=[idx_to_class[i] for i in range(len(idx_to_class))]
)

## ======================
## 4. Per-Class Accuracy
## ======================
def per_class_accuracy(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.bar(classes, class_acc)
    plt.xticks(rotation=45)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('per_class_accuracy.png')
    plt.show()
    
    print("\n=== Per-Class Accuracy ===")
    for i, acc in enumerate(class_acc):
        print(f"{classes[i]}: {acc:.4f}")

per_class_accuracy(
    y_test, 
    y_pred, 
    classes=[idx_to_class[i] for i in range(len(idx_to_class))]
)

## ======================
## 5. ROC Curve (for binary/multi-class)
## ======================
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

def plot_roc_curve(y_true, y_pred_prob, classes):
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=np.arange(len(classes)))
    
    # Compute ROC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(
            fpr[i], 
            tpr[i], 
            color=color,
            label=f'ROC {classes[i]} (AUC = {roc_auc[i]:.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Multi-Class)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()

plot_roc_curve(
    y_test, 
    y_pred_prob, 
    classes=[idx_to_class[i] for i in range(len(idx_to_class))]
)

## ======================
## 6. Top-K Accuracy (Optional)
## ======================
def top_k_accuracy(y_true, y_pred_prob, k=3):
    top_k = np.argsort(y_pred_prob, axis=1)[:, -k:]
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k[i]:
            correct += 1
    return correct / len(y_true)

top3_acc = top_k_accuracy(y_test, y_pred_prob, k=3)
print(f"\nTop-3 Accuracy: {top3_acc:.4f}")