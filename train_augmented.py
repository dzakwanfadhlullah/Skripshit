"""
Phase 5: Training Skenario 2 - Model Robust (Low-Light Augmented)
Hyperparameter IDENTIK dengan baseline agar perbandingan adil.
Jalankan di Google Colab dengan GPU.
"""
from ultralytics import YOLO
import os

def train_augmented():
    """
    Training YOLOv8 dengan dataset campuran (original + low-light augmented).
    Semua hyperparameter SAMA PERSIS dengan baseline training.
    """
    # Load model pretrained BARU (fresh start, bukan lanjutan baseline)
    model = YOLO("yolov8n.pt")

    print("\n--- Memulai Training Skenario 2: Augmented Low-Light ---")
    results = model.train(
        data="skripshit_data.yaml",  # YAML yang sama, sekarang train mengandung low_ files
        epochs=50,                    # SAMA dengan baseline
        imgsz=640,                    # SAMA dengan baseline
        batch=16,                     # SAMA dengan baseline
        name="yolov8_augmented",      # Nama berbeda untuk membedakan hasil
        device=0,                     # GPU
        save=True,
        project="runs/detect",

        # --- Hyperparameter IDENTIK dengan baseline ---
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        mosaic=1.0,
        mixup=0.0,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        erasing=0.4,
        copy_paste=0.0,

        # --- Monitoring ---
        plots=True,                   # Grafik loss otomatis
        val=True,                     # Evaluasi setiap epoch
        patience=100,                 # Early stopping patience
        seed=0,                       # Reprodusibilitas
        deterministic=True,
    )

    print("\nTraining Selesai!")
    print(f"Weights terbaik: {os.path.abspath('runs/detect/yolov8_augmented/weights/best.pt')}")
    return results

if __name__ == "__main__":
    train_augmented()
