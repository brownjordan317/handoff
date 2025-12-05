#!/usr/bin/env python3
import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

# === CONFIG ===
DATASET_PATH = "/home/megrad/Documents/asl_alphabet_LO"
MODEL_PATH = "asl_letters_only.task"
FPS = 30
SCORE_THRESHOLD = 0.7
OUTPUT_CM_PNG = "confusion_matrix.png"
OUTPUT_JSON = "gesture_results.json"

# === MEDIA PIPE SETUP ===
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,  # IMAGE mode for single images
    num_hands=2,
)

gesture_recognizer = GestureRecognizer.create_from_options(gesture_options)

# === LOAD DATASET ===
labels = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
labels.sort()  # ensure consistent ordering

y_true = []
y_pred = []

# === EVALUATE EVERY IMAGE ===
print("[INFO] Evaluating gestures on dataset...")
total_images = sum([len([f for f in os.listdir(os.path.join(DATASET_PATH, l)) if f.lower().endswith((".jpg", ".png"))])
                    for l in labels])

with tqdm(total=total_images, desc="Processing images") as pbar:
    for label_idx, label_name in enumerate(labels):
        label_folder = os.path.join(DATASET_PATH, label_name)
        for filename in os.listdir(label_folder):
            if not filename.lower().endswith((".jpg", ".png")):
                continue
            img_path = os.path.join(label_folder, filename)
            frame = cv2.imread(img_path)
            if frame is None:
                pbar.update(1)
                continue

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Recognize gestures
            result = gesture_recognizer.recognize(mp_image)
            predicted_label = None

            # Take highest scoring hand gesture above threshold
            if result.gestures and result.handedness:
                scores = []
                names = []
                for gesture_list in result.gestures:
                    if gesture_list:
                        gesture = gesture_list[0]
                        if gesture.score >= SCORE_THRESHOLD:
                            scores.append(gesture.score)
                            names.append(gesture.category_name)
                if scores:
                    predicted_label = names[np.argmax(scores)]

            # Fallback if no gesture passed threshold
            if predicted_label is None:
                predicted_label = "none"

            y_true.append(label_idx)
            y_pred.append(labels.index(predicted_label) if predicted_label in labels else -1)
            pbar.update(1)

# === CONFUSION MATRIX ===
cm_labels = labels if -1 in y_pred else labels
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))) + ([-1] if -1 in y_pred else []))

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=cm_labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Gesture Recognition Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_CM_PNG)
print(f"[INFO] Confusion matrix saved to {OUTPUT_CM_PNG}")
plt.close()

# === CLASSIFICATION REPORT ===
y_pred_str = [labels[i] if i >= 0 else "none" for i in y_pred]
report_dict = classification_report([labels[i] for i in y_true], y_pred_str, target_names=cm_labels, output_dict=True)

# Save to JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(report_dict, f, indent=4)
print(f"[INFO] Classification results saved to {OUTPUT_JSON}")
