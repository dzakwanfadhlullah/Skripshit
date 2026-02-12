"""
Phase 6.2: Evaluate Robustness — 4-Way Test
Menguji 2 model (Baseline & Augmented) pada 2 kondisi (Normal & Low-Light).
Semua dijalankan secara otomatis dan hasilnya disimpan ke CSV.
"""
import os
import yaml
import csv
from ultralytics import YOLO


# ==================== KONFIGURASI ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "skripshit_yolo")

BASELINE_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "yolov8_baseline_final", "weights", "best.pt")
AUGMENTED_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "runs", "detect", "yolov8_augmented", "weights", "best.pt")

OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results")
# =====================================================


def create_eval_yaml(val_folder_name, yaml_name):
    """Buat file YAML sementara yang mengarah ke folder val tertentu."""
    yaml_path = os.path.join(BASE_DIR, yaml_name)
    data = {
        "path": os.path.abspath(DATASET_DIR),
        "train": "images/train",
        "val": f"images/{val_folder_name}",
        "names": {0: "face"}
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return yaml_path


def run_evaluation(weights_path, yaml_path, test_name):
    """Jalankan evaluasi model dan return metrik."""
    print(f"\n{'='*60}")
    print(f"  TEST: {test_name}")
    print(f"  Weights: {os.path.basename(os.path.dirname(os.path.dirname(weights_path)))}")
    print(f"  Val set: {yaml_path}")
    print(f"{'='*60}\n")

    model = YOLO(weights_path)
    results = model.val(
        data=yaml_path,
        imgsz=640,
        batch=16,
        device=0,
        plots=True,
        save_json=False,
        project=OUTPUT_DIR,
        name=test_name,
    )

    metrics = {
        "test_name": test_name,
        "precision": round(results.box.mp, 5),
        "recall": round(results.box.mr, 5),
        "mAP50": round(results.box.map50, 5),
        "mAP50-95": round(results.box.map, 5),
    }

    print(f"\n--- Hasil {test_name} ---")
    for k, v in metrics.items():
        if k != "test_name":
            print(f"  {k}: {v}")

    return metrics


def print_comparison_table(all_results):
    """Cetak tabel perbandingan akhir."""
    print(f"\n{'='*70}")
    print(f"  HASIL EVALUASI ROBUSTNESS — PERBANDINGAN LENGKAP")
    print(f"{'='*70}")
    print(f"{'Test':<30} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    print(f"{'-'*70}")
    for r in all_results:
        print(f"{r['test_name']:<30} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['mAP50']:>10.4f} {r['mAP50-95']:>10.4f}")
    print(f"{'-'*70}")

    # Hitung robustness drop
    baseline_normal = next(r for r in all_results if r["test_name"] == "baseline_normal")
    baseline_lowlight = next(r for r in all_results if r["test_name"] == "baseline_lowlight")
    augmented_normal = next(r for r in all_results if r["test_name"] == "augmented_normal")
    augmented_lowlight = next(r for r in all_results if r["test_name"] == "augmented_lowlight")

    drop_baseline = (baseline_normal["mAP50"] - baseline_lowlight["mAP50"]) / baseline_normal["mAP50"] * 100
    drop_augmented = (augmented_normal["mAP50"] - augmented_lowlight["mAP50"]) / augmented_normal["mAP50"] * 100

    print(f"\n📊 ANALISIS ROBUSTNESS (mAP50):")
    print(f"  Baseline  : Normal {baseline_normal['mAP50']:.4f} → Low-Light {baseline_lowlight['mAP50']:.4f} | Drop: {drop_baseline:.2f}%")
    print(f"  Augmented : Normal {augmented_normal['mAP50']:.4f} → Low-Light {augmented_lowlight['mAP50']:.4f} | Drop: {drop_augmented:.2f}%")
    print(f"\n  🔑 Improvement: Augmented drop {drop_augmented:.2f}% vs Baseline drop {drop_baseline:.2f}%")

    if drop_augmented < drop_baseline:
        print(f"  ✅ Model Augmented LEBIH ROBUST ({drop_baseline - drop_augmented:.2f}% lebih tahan)")
    else:
        print(f"  ⚠️  Model Augmented tidak lebih robust dari baseline")

    return {
        "drop_baseline": drop_baseline,
        "drop_augmented": drop_augmented,
    }


def save_results_csv(all_results, drops):
    """Simpan hasil ke CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "robustness_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n📁 Hasil disimpan ke: {csv_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Cek weights
    for name, path in [("Baseline", BASELINE_WEIGHTS), ("Augmented", AUGMENTED_WEIGHTS)]:
        if not os.path.exists(path):
            print(f"❌ {name} weights tidak ditemukan: {path}")
            return
        print(f"✅ {name} weights: {path}")

    # Buat YAML untuk kedua val set
    yaml_normal = create_eval_yaml("val", "eval_normal.yaml")
    yaml_lowlight = create_eval_yaml("val_lowlight", "eval_lowlight.yaml")

    # Jalankan 4 test
    all_results = []
    all_results.append(run_evaluation(BASELINE_WEIGHTS, yaml_normal, "baseline_normal"))
    all_results.append(run_evaluation(BASELINE_WEIGHTS, yaml_lowlight, "baseline_lowlight"))
    all_results.append(run_evaluation(AUGMENTED_WEIGHTS, yaml_normal, "augmented_normal"))
    all_results.append(run_evaluation(AUGMENTED_WEIGHTS, yaml_lowlight, "augmented_lowlight"))

    # Perbandingan
    drops = print_comparison_table(all_results)
    save_results_csv(all_results, drops)

    # Cleanup YAML sementara
    os.remove(yaml_normal)
    os.remove(yaml_lowlight)

    print("\n🎉 Evaluasi selesai!")


if __name__ == "__main__":
    main()
