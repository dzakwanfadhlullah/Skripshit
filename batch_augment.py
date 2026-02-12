
import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_brightness(image, value=-50):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.int16) + value, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def augment_to_low_light(image, gamma=0.4, brightness=-30):
    img_low = adjust_gamma(image, gamma=gamma)
    img_low = adjust_brightness(img_low, value=brightness)
    return img_low

def batch_augment(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, gamma=0.4, brightness=-30):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('low_')]
    print(f"=== BATCH LOW-LIGHT AUGMENTATION (RESUME MODE) ===")
    success, already_done, skipped = 0, 0, 0
    for img_name in tqdm(img_files, desc="Processing"):
        base_name = os.path.splitext(img_name)[0]
        ext = os.path.splitext(img_name)[1]
        dst_img_path = os.path.join(dst_img_dir, f"low_{base_name}{ext}")
        dst_lbl_path = os.path.join(dst_lbl_dir, f"low_{base_name}.txt")
        if os.path.exists(dst_img_path):
            already_done += 1
            continue
        src_img_path = os.path.join(src_img_dir, img_name)
        src_lbl_path = os.path.join(src_lbl_dir, base_name + ".txt")
        if not os.path.exists(src_lbl_path):
            skipped += 1
            continue
        img = cv2.imread(src_img_path)
        if img is None:
            skipped += 1
            continue
        img_low = augment_to_low_light(img, gamma=gamma, brightness=brightness)
        cv2.imwrite(dst_img_path, img_low)
        shutil.copy2(src_lbl_path, dst_lbl_path)
        success += 1
    print(f"\nBerhasil: {success} | Melewati (Sudah ada): {already_done} | Error: {skipped}")

def validate_augmentation(orig_img_dir, aug_img_dir, label_dir, num_samples=4):
    orig_files = [f for f in os.listdir(orig_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('low_')]
    samples = random.sample(orig_files, min(num_samples, len(orig_files)))
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    for i, img_name in enumerate(samples):
        base_name, ext = os.path.splitext(img_name)
        orig_img = cv2.imread(os.path.join(orig_img_dir, img_name))
        aug_img = cv2.imread(os.path.join(aug_img_dir, f"low_{base_name}{ext}"))
        axes[i, 0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        if aug_img is not None:
            axes[i, 1].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title("Low-Light Augmented")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()
