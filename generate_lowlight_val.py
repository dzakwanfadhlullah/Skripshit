"""
Phase 6.1: Generate Low-Light Validation Set
Membuat versi low-light dari semua gambar val untuk pengujian robustness.
Parameter augmentasi IDENTIK dengan Phase 4 (gamma=0.4, brightness=-30).
"""
import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm


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


def generate_lowlight_val(base_dir, gamma=0.4, brightness=-30):
    """
    Membuat folder val_lowlight berisi versi gelap dari semua gambar val.
    Label di-copy karena bounding box tidak berubah.
    """
    val_img = os.path.join(base_dir, "images", "val")
    val_lbl = os.path.join(base_dir, "labels", "val")
    low_img = os.path.join(base_dir, "images", "val_lowlight")
    low_lbl = os.path.join(base_dir, "labels", "val_lowlight")

    os.makedirs(low_img, exist_ok=True)
    os.makedirs(low_lbl, exist_ok=True)

    img_files = [f for f in os.listdir(val_img)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"=== GENERATE LOW-LIGHT VAL SET ===")
    print(f"Source       : {val_img}")
    print(f"Destination  : {low_img}")
    print(f"Total images : {len(img_files)}")
    print(f"Gamma        : {gamma}")
    print(f"Brightness   : {brightness}")
    print(f"==================================\n")

    success, skipped = 0, 0
    for img_name in tqdm(img_files, desc="Generating low-light val"):
        base_name = os.path.splitext(img_name)[0]
        ext = os.path.splitext(img_name)[1]

        dst_img_path = os.path.join(low_img, img_name)
        dst_lbl_path = os.path.join(low_lbl, base_name + ".txt")
        src_lbl_path = os.path.join(val_lbl, base_name + ".txt")

        # Skip jika sudah ada
        if os.path.exists(dst_img_path):
            success += 1
            continue

        # Baca dan augmentasi
        img = cv2.imread(os.path.join(val_img, img_name))
        if img is None:
            skipped += 1
            continue

        # Terapkan augmentasi (identik dengan Phase 4)
        img_low = adjust_gamma(img, gamma=gamma)
        img_low = adjust_brightness(img_low, value=brightness)

        # Simpan
        cv2.imwrite(dst_img_path, img_low)

        # Copy label (bounding box tidak berubah)
        if os.path.exists(src_lbl_path):
            shutil.copy2(src_lbl_path, dst_lbl_path)

        success += 1

    print(f"\n✅ Selesai!")
    print(f"Low-light val images : {len(os.listdir(low_img))}")
    print(f"Low-light val labels : {len(os.listdir(low_lbl))}")
    print(f"Skipped              : {skipped}")


if __name__ == "__main__":
    BASE = "datasets/skripshit_yolo"
    generate_lowlight_val(BASE, gamma=0.4, brightness=-30)
