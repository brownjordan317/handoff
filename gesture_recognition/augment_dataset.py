import os
import cv2
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Root dataset folder
DATASET_DIR = "/home/megrad/Documents/asl_alphabet_LO"

# Define torchvision augmentations (without homography first)
base_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
])

# Helper: Apply random homography warp using OpenCV
def random_homography(image, max_warp=0.1):
    """Apply random homography warp to a PIL image."""
    img = np.array(image)
    h, w = img.shape[:2]

    # Original corners
    pts1 = np.float32([[0,0],[w,0],[w,h],[0,h]])

    # Perturb corners slightly
    shift = max_warp * min(h, w)
    pts2 = pts1 + np.random.uniform(-shift, shift, pts1.shape).astype(np.float32)

    # Compute homography and warp
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (w,h), borderMode=cv2.BORDER_REFLECT101)

    return Image.fromarray(warped)

# Augmentation pipeline
def augment_image(img):
    # Apply torchvision transforms
    img = base_transform(img)

    # Apply homography with 50% chance
    if random.random() > 0.5:
        img = random_homography(img)

    return img

# Function to process a single image (parallelized)
def process_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        save_dir = os.path.dirname(img_path)

        results = []
        for i in range(1, 5):  # 4 augmentations
            aug_img = augment_image(img)
            save_name = f"augmented_{base_name}_{i}{ext}"
            save_path = os.path.join(save_dir, save_name)
            aug_img.save(save_path, quality=80)  # lower quality = faster
            results.append(save_path)
        return results
    except Exception as e:
        return f"Error processing {img_path}: {e}"

if __name__ == "__main__":
    # Collect all image paths
    all_images = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(root, file))

    print(f"Found {len(all_images)} images. Augmenting...")

    # Use multiprocessing pool
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_image, all_images), total=len(all_images)))
