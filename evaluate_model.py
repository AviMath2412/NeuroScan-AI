import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def evaluate_model(data_dir="data", model_path="best_model.pth", batch_size=32):
    """
    Evaluates the trained model on the testing set and calculates:
    - Overall accuracy
    - Per-class True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
    - Per-class False Positive Rate (FPR)
    - Binary screening False Positives (Healthy scan predicted as Tumor)
    - Full Confusion Matrix
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    test_dir = os.path.join(data_dir, "Testing")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    class_names = test_dataset.classes
    num_classes = len(class_names)
    total_samples = len(test_dataset)

    # Initialize model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    print(f"Evaluating {total_samples} test images across classes: {class_names}...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 1. Overall Accuracy
    correct = np.sum(all_preds == all_targets)
    overall_accuracy = float(correct / total_samples)

    # 2. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)

    # 3. Per-class metrics: TP, FP, FN, TN, FPR
    per_class_metrics = {}
    for i, cls_name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        tn = int(total_samples - (tp + fp + fn))
        
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        per_class_metrics[cls_name] = {
            "true_positives (TP)": tp,
            "false_positives (FP)": fp,
            "false_negatives (FN)": fn,
            "true_negatives (TN)": tn,
            "false_positive_rate (FPR)": round(fpr, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        }

    # 4. Binary Tumor Screening False Positives
    # In clinical practice: Healthy ('notumor') flagged as having a tumor
    binary_screening = {}
    if 'notumor' in class_names:
        notumor_idx = class_names.index('notumor')
        # Total healthy patients
        total_healthy = int(np.sum(all_targets == notumor_idx))
        # Healthy patients diagnosed as any tumor
        healthy_false_positives = int(np.sum((all_targets == notumor_idx) & (all_preds != notumor_idx)))
        healthy_true_negatives = total_healthy - healthy_false_positives
        screening_fpr = float(healthy_false_positives / total_healthy) if total_healthy > 0 else 0.0

        # Patients with a tumor missed (False Negative)
        total_tumor_cases = total_samples - total_healthy
        tumor_missed_as_healthy = int(np.sum((all_targets != notumor_idx) & (all_preds == notumor_idx)))

        binary_screening = {
            "healthy_scans_total": total_healthy,
            "healthy_misdiagnosed_as_tumor (False Positives)": healthy_false_positives,
            "healthy_correctly_identified (True Negatives)": healthy_true_negatives,
            "screening_false_positive_rate": round(screening_fpr, 4),
            "tumor_cases_total": total_tumor_cases,
            "tumor_missed_as_healthy (False Negatives)": tumor_missed_as_healthy
        }

    results = {
        "total_samples": total_samples,
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_accuracy_percentage": f"{overall_accuracy * 100:.2f}%",
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "per_class_metrics": per_class_metrics,
        "binary_screening_metrics": binary_screening
    }

    # Print a clean report to stdout
    print("\n" + "=" * 65)
    print(f"            MODEL ACCURACY & PERFORMANCE REPORT            ")
    print("=" * 65)
    print(f"Total Test Samples: {total_samples}")
    print(f"Overall Accuracy  : {overall_accuracy * 100:.2f}% ({correct}/{total_samples} correct)\n")

    print("--- PER-CLASS FALSE POSITIVES & METRICS ---")
    print(f"{'Class':<14} | {'TP':<5} | {'FP (False Pos)':<15} | {'FN':<5} | {'FPR':<8} | {'Precision':<9} | {'Recall':<8}")
    print("-" * 75)
    for cls_name, m in per_class_metrics.items():
        print(f"{cls_name:<14} | {m['true_positives (TP)']:<5} | {m['false_positives (FP)']:<15} | {m['false_negatives (FN)']:<5} | {m['false_positive_rate (FPR)']:<8.4f} | {m['precision']:<9.4f} | {m['recall']:<8.4f}")

    if binary_screening:
        print("\n--- CLINICAL SCREENING (TUMOR VS HEALTHY) ---")
        print(f"Total Healthy ('notumor') Scans        : {binary_screening['healthy_scans_total']}")
        print(f"False Positives (Healthy flagged as Tumor): {binary_screening['healthy_misdiagnosed_as_tumor (False Positives)']}")
        print(f"Screening False Positive Rate (FPR)       : {binary_screening['screening_false_positive_rate'] * 100:.2f}%")
        print(f"Tumors Missed as Healthy (False Negatives): {binary_screening['tumor_missed_as_healthy (False Negatives)']}")

    print("\n--- CONFUSION MATRIX ---")
    header = f"{'True \\ Pred':<14}" + "".join([f"{name[:8]:>10}" for name in class_names])
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{name:<14}" + "".join([f"{cm[i, j]:>10}" for j in range(num_classes)])
        print(row)
    print("=" * 65 + "\n")

    return results

if __name__ == "__main__":
    evaluate_model()
