import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F

def save_evaluation_to_text(predictions, num_labels: int, results_path: Path, model_name: str):

    if num_labels==5:

    # === Class ID to Name Mapping (IDs are +1 from model index) ===
        class_names = {
            1: "Acropora Branching",
            2: "Non-acropora Massive",
            3: "Other Corals",
            4: "Sand",
            5: "Seagrass"
        }
    elif num_labels==4:

        class_names = {
            1: "Acropora Branching",
            2: "Other Corals",
            3: "Non-acropora Massive",
            4: "Sand"
        }
    else:
        raise ValueError(f"Unexpected number of labels: {num_labels}. Expected 4 or 5.")

    # === Extract and align predictions ===
    logits = torch.from_numpy(predictions.predictions)  # shape: [B, C, H, W]
    labels = torch.from_numpy(predictions.label_ids)    # shape: [B, H, W]
    logits_resized = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

    pred = torch.argmax(logits_resized, dim=1).flatten().numpy()
    gt = labels.flatten().numpy()

    # === Compute metrics ===
    labels_set = sorted(set(np.unique(gt)) | set(np.unique(pred)))
    cm = confusion_matrix(gt, pred, labels=labels_set)

    epsilon = 1e-7
    intersection = np.diag(cm)
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - intersection
    iou_per_class = intersection / (union + epsilon)
    mean_iou = np.mean(iou_per_class)
    pixel_acc = np.sum(intersection) / np.sum(cm)

    # === Map class ID to name (+1 to class index) ===
    iou_dict = {
        class_names[label + 1]: round(float(iou), 4)
        for label, iou in zip(labels_set, iou_per_class)
        if label + 1 in class_names
    }

    # === Load existing results if any ===
    results_dict = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results_dict = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Warning: Failed to decode existing results file. Starting fresh.")

    # === Store results for this model ===
    results_dict[model_name] = {
        "pixel_acc": round(float(pixel_acc), 4),
        "mean_iou": round(float(mean_iou), 4),
        "iou_per_class": iou_dict
    }

    print(f"\n📈 Results for model {model_name}:"
        f"\n- Pixel Accuracy: {results_dict[model_name]['pixel_acc']}"
        f"\n- Mean IoU: {results_dict[model_name]['mean_iou']}"
        f"\n- IoU per class: {results_dict[model_name]['iou_per_class']}")
    # === Save updated results ===
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"✅ Results saved to {results_path}")