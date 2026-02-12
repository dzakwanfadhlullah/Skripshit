"""
Phase 5 Preparation: Split Validation Set
Memisahkan ~20% data ORIGINAL (bukan low_) ke folder val.
Jalankan di Google Colab.
"""
import os
import shutil
import random

def split_val(base_dir, val_ratio=0.2, seed=42):
    """
    Memindahkan sebagian data original dari train ke val.
    Hanya data ORIGINAL (tanpa prefix 'low_') yang dipindahkan.
    """
    random.seed(seed)

    train_img = os.path.join(base_dir, "images", "train")
    train_lbl = os.path.join(base_dir, "labels", "train")
    val_img   = os.path.join(base_dir, "images", "val")
    val_lbl   = os.path.join(base_dir, "labels", "val")

    os.makedirs(val_img, exist_ok=True)
    os.makedirs(val_lbl, exist_ok=True)

    # Ambil hanya file ORIGINAL (bukan low_)
    all_imgs = [f for f in os.listdir(train_img)
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('low_')]

    # Hanya ambil yang punya label
    valid_imgs = []
    for f in all_imgs:
        base = os.path.splitext(f)[0]
        if os.path.exists(os.path.join(train_lbl, base + ".txt")):
            valid_imgs.append(f)

    random.shuffle(valid_imgs)
    num_val = int(len(valid_imgs) * val_ratio)
    val_set = valid_imgs[:num_val]

    print(f"=== SPLIT VALIDATION SET ===")
    print(f"Total original images : {len(valid_imgs)}")
    print(f"Val set (20%)         : {num_val}")
    print(f"Train set (80%)       : {len(valid_imgs) - num_val}")
    print(f"============================\n")

    moved = 0
    for img_name in val_set:
        base = os.path.splitext(img_name)[0]
        ext  = os.path.splitext(img_name)[1]

        # Pindahkan gambar original
        shutil.move(os.path.join(train_img, img_name),
                    os.path.join(val_img, img_name))

        # Pindahkan label original
        shutil.move(os.path.join(train_lbl, base + ".txt"),
                    os.path.join(val_lbl, base + ".txt"))

        # Hapus juga versi low-light dari train (tidak boleh ada di train jika original-nya di val)
        low_img = os.path.join(train_img, f"low_{base}{ext}")
        low_lbl = os.path.join(train_lbl, f"low_{base}.txt")
        if os.path.exists(low_img):
            os.remove(low_img)
        if os.path.exists(low_lbl):
            os.remove(low_lbl)

        moved += 1

    print(f"Berhasil dipindahkan ke val: {moved} pasang (image + label)")
    print(f"Isi val images : {len(os.listdir(val_img))}")
    print(f"Isi val labels : {len(os.listdir(val_lbl))}")
    print(f"Sisa train images : {len(os.listdir(train_img))}")
    print(f"Sisa train labels : {len(os.listdir(train_lbl))}")
