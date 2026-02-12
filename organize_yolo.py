
import os
import shutil
import random

def organize_yolo(source_base_dir, project_base_dir, split_ratio=(0.8, 0.2)):
    print(f"\n--- Memulai Organisasi YOLO (Deep Search) ---")
    target_structure = [
        "datasets/skripshit_yolo/images/train",
        "datasets/skripshit_yolo/images/val",
        "datasets/skripshit_yolo/labels/train",
        "datasets/skripshit_yolo/labels/val"
    ]
    for folder in target_structure:
        os.makedirs(os.path.join(project_base_dir, folder), exist_ok=True)

    all_images = {}
    all_labels = {}

    print(f"Mencari file di {source_base_dir}...")
    for root, dirs, files in os.walk(source_base_dir):
        for f in files:
            name_no_ext = os.path.splitext(f)[0]
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images[name_no_ext] = os.path.join(root, f)
            elif f.lower().endswith('.txt') and name_no_ext.lower() not in ['classes', 'notes', 'readme']:
                all_labels[name_no_ext] = os.path.join(root, f)

    matched_pairs = []
    for name in all_images:
        if name in all_labels:
            matched_pairs.append((all_images[name], all_labels[name]))

    print(f"Ditemukan: {len(all_images)} Gambar, {len(all_labels)} Label.")
    print(f"BERHASIL DIPASANGKAN: {len(matched_pairs)} pasang.")
    
    if not matched_pairs:
        print("GAGAL: Tidak ada pasangan yang cocok. Cek log pencarian di atas.")
        return

    random.shuffle(matched_pairs)
    train_count = int(len(matched_pairs) * split_ratio[0])
    
    def move_files(pairs, subset):
        for img_path, lbl_path in pairs:
            img_ext = os.path.splitext(img_path)[1]
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            shutil.copy2(img_path, os.path.join(project_base_dir, f"datasets/skripshit_yolo/images/{subset}", base_name + img_ext))
            shutil.copy2(lbl_path, os.path.join(project_base_dir, f"datasets/skripshit_yolo/labels/{subset}", base_name + ".txt"))

    move_files(matched_pairs[:train_count], "train")
    move_files(matched_pairs[train_count:], "val")

    yaml_content = f"path: ../datasets/skripshit_yolo\ntrain: images/train\nval: images/val\n\nnames:\n  0: face\n"
    with open(os.path.join(project_base_dir, "skripshit_data.yaml"), "w") as f:
        f.write(yaml_content)
    print("SELESAI! File skripshit_data.yaml telah dibuat.")
