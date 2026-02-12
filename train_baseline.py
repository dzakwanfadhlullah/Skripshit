from ultralytics import YOLO
import os

def train_baseline():
    """
    Melatih model YOLOv8 baseline menggunakan data normal.
    """
    # 1. Load model pretrained (YOLOv8 Nano - ringan dan cepat untuk skripsi)
    model = YOLO("yolov8n.pt") 

    # 2. Mulai Training
    print("\n--- Memulai Training Baseline (Skenario 1) ---")
    results = model.train(
        data="skripshit_data.yaml", 
        epochs=50,                  # Standar training yang baik untuk Baseline
        imgsz=640, 
        batch=16, 
        name="yolov8_baseline",
        device=0,
        save=True,                  # Simpan weight terbaik
        project="runs/detect"       # Lokasi output
    )
    
    print("\nTraining Selesai!")
    print(f"Weights terbaik disimpan di: {os.path.abspath('runs/detect/yolov8_baseline/weights/best.pt')}")

if __name__ == "__main__":
    # train_baseline()
    pass
