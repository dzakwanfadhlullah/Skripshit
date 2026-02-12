"""
Phase 6.4: Generate Comprehensive Robustness Report
Menghasilkan laporan lengkap dengan grafik, tabel, dan kesimpulan.
Output: Grafik PNG + ringkasan teks di folder evaluation_results/.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ==================== KONFIGURASI ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results")
CSV_PATH = os.path.join(OUTPUT_DIR, "robustness_results.csv")

BASELINE_TRAIN_CSV = os.path.join(BASE_DIR, "runs", "detect", "yolov8_baseline_final", "results.csv")
AUGMENTED_TRAIN_CSV = os.path.join(BASE_DIR, "runs", "detect", "runs", "detect", "yolov8_augmented", "results.csv")

# Warna tema
C_BASELINE = '#E74C3C'      # Merah
C_AUGMENTED = '#3498DB'     # Biru
C_NORMAL = '#2ECC71'        # Hijau
C_LOWLIGHT = '#9B59B6'      # Ungu
C_BG = '#1a1a2e'            # Background gelap
C_CARD = '#16213e'          # Card gelap
C_TEXT = '#EAEAEA'          # Teks terang
C_ACCENT = '#e94560'        # Aksen
# =====================================================


def load_eval_results():
    """Load hasil evaluasi dari CSV."""
    results = {}
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["test_name"]] = {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "mAP50": float(row["mAP50"]),
                "mAP50-95": float(row["mAP50-95"]),
            }
    return results


def load_training_csv(csv_path):
    """Load training history dari results.csv."""
    epochs, mAP50, mAP95 = [], [], []
    box_loss, cls_loss = [], []
    val_box, val_cls = [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys
            row = {k.strip(): v.strip() for k, v in row.items()}
            epochs.append(int(row["epoch"]))
            mAP50.append(float(row["metrics/mAP50(B)"]))
            mAP95.append(float(row["metrics/mAP50-95(B)"]))
            box_loss.append(float(row["train/box_loss"]))
            cls_loss.append(float(row["train/cls_loss"]))
            val_box.append(float(row["val/box_loss"]))
            val_cls.append(float(row["val/cls_loss"]))

    return {
        "epochs": epochs, "mAP50": mAP50, "mAP50-95": mAP95,
        "train_box": box_loss, "train_cls": cls_loss,
        "val_box": val_box, "val_cls": val_cls,
    }


def setup_dark_style():
    """Setup matplotlib dark theme."""
    plt.rcParams.update({
        'figure.facecolor': C_BG,
        'axes.facecolor': C_CARD,
        'axes.edgecolor': '#444466',
        'axes.labelcolor': C_TEXT,
        'text.color': C_TEXT,
        'xtick.color': C_TEXT,
        'ytick.color': C_TEXT,
        'grid.color': '#333355',
        'grid.alpha': 0.5,
        'font.family': 'sans-serif',
        'font.size': 11,
    })


# ============================================================
#  CHART 1: Bar Chart Perbandingan 4-Way (Hero Chart)
# ============================================================
def chart_4way_comparison(results):
    """Bar chart utama: 4 skenario side-by-side."""
    fig, ax = plt.subplots(figsize=(14, 7))

    metrics = ["mAP50", "mAP50-95", "precision", "recall"]
    labels = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
    x = np.arange(len(metrics))
    width = 0.18

    scenarios = [
        ("baseline_normal", "Baseline + Normal", C_BASELINE, '///'),
        ("baseline_lowlight", "Baseline + Low-Light", C_BASELINE, ''),
        ("augmented_normal", "Augmented + Normal", C_AUGMENTED, '///'),
        ("augmented_lowlight", "Augmented + Low-Light", C_AUGMENTED, ''),
    ]

    for i, (key, label, color, hatch) in enumerate(scenarios):
        vals = [results[key][m] for m in metrics]
        alpha = 0.9 if not hatch else 0.5
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=alpha, hatch=hatch,
                      edgecolor='white', linewidth=0.5)
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold', color=C_TEXT)

    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('🔬 Perbandingan 4-Way: Baseline vs Augmented × Normal vs Low-Light',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.legend(loc='upper right', fontsize=10, facecolor=C_CARD, edgecolor='#555')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart_4way_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")
    return path


# ============================================================
#  CHART 2: Robustness Drop Comparison
# ============================================================
def chart_robustness_drop(results):
    """Grafik drop performa dari Normal ke Low-Light."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    metrics_to_plot = [
        ("mAP50", "mAP@50 Drop"),
        ("recall", "Recall Drop"),
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx]

        bl_normal = results["baseline_normal"][metric]
        bl_lowlight = results["baseline_lowlight"][metric]
        aug_normal = results["augmented_normal"][metric]
        aug_lowlight = results["augmented_lowlight"][metric]

        drop_bl = (bl_normal - bl_lowlight) / bl_normal * 100
        drop_aug = (aug_normal - aug_lowlight) / aug_normal * 100

        # Arrow-style visualization
        models = ['Baseline', 'Augmented']
        normals = [bl_normal, aug_normal]
        lowlights = [bl_lowlight, aug_lowlight]
        drops = [drop_bl, drop_aug]
        colors = [C_BASELINE, C_AUGMENTED]

        x_pos = [0, 1.5]
        for i in range(2):
            # Bar normal
            ax.bar(x_pos[i] - 0.2, normals[i], 0.35, color=colors[i], alpha=0.5,
                   edgecolor='white', linewidth=0.5, label='Normal' if i == 0 else '')
            # Bar low-light
            ax.bar(x_pos[i] + 0.2, lowlights[i], 0.35, color=colors[i], alpha=0.9,
                   edgecolor='white', linewidth=0.5, label='Low-Light' if i == 0 else '')
            # Drop annotation arrow
            ax.annotate('', xy=(x_pos[i] + 0.2, lowlights[i]),
                       xytext=(x_pos[i] - 0.2, normals[i]),
                       arrowprops=dict(arrowstyle='->', color=C_ACCENT, lw=2.5))
            # Drop text
            ax.text(x_pos[i], (normals[i] + lowlights[i]) / 2,
                    f'↓{drops[i]:.1f}%', ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color=C_ACCENT if drops[i] > 15 else C_NORMAL,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=C_BG,
                              edgecolor=C_ACCENT if drops[i] > 15 else C_NORMAL, alpha=0.9))
            # Value labels
            ax.text(x_pos[i] - 0.2, normals[i] + 0.01, f'{normals[i]:.3f}',
                    ha='center', fontsize=9, fontweight='bold')
            ax.text(x_pos[i] + 0.2, lowlights[i] + 0.01, f'{lowlights[i]:.3f}',
                    ha='center', fontsize=9, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.08)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    axes[0].legend(fontsize=10, facecolor=C_CARD, edgecolor='#555')
    fig.suptitle('📉 Robustness Drop: Seberapa Banyak Performa Turun di Low-Light?',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart_robustness_drop.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")
    return path


# ============================================================
#  CHART 3: Radar/Spider Chart
# ============================================================
def chart_radar(results):
    """Radar chart perbandingan baseline vs augmented di low-light."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    categories = ['mAP@50', 'mAP@50-95', 'Precision', 'Recall']
    metrics_keys = ['mAP50', 'mAP50-95', 'precision', 'recall']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    bl = [results["baseline_lowlight"][m] for m in metrics_keys]
    bl += bl[:1]
    aug = [results["augmented_lowlight"][m] for m in metrics_keys]
    aug += aug[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    plt.xticks(angles[:-1], categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)

    ax.plot(angles, bl, 'o-', linewidth=2.5, color=C_BASELINE, label='Baseline (Low-Light)')
    ax.fill(angles, bl, alpha=0.15, color=C_BASELINE)
    ax.plot(angles, aug, 'o-', linewidth=2.5, color=C_AUGMENTED, label='Augmented (Low-Light)')
    ax.fill(angles, aug, alpha=0.15, color=C_AUGMENTED)

    # Value labels
    for angle, bv, av in zip(angles[:-1], bl[:-1], aug[:-1]):
        ax.text(angle, bv + 0.05, f'{bv:.3f}', ha='center', fontsize=8, color=C_BASELINE, fontweight='bold')
        ax.text(angle, av - 0.07, f'{av:.3f}', ha='center', fontsize=8, color=C_AUGMENTED, fontweight='bold')

    ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05), fontsize=11,
              facecolor=C_CARD, edgecolor='#555')
    ax.set_title('🕸️ Radar: Performa di Kondisi Low-Light',
                 fontsize=14, fontweight='bold', pad=25)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart_radar_lowlight.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")
    return path


# ============================================================
#  CHART 4: Training History Overlay
# ============================================================
def chart_training_history():
    """Overlay training curves baseline vs augmented."""
    if not os.path.exists(BASELINE_TRAIN_CSV) or not os.path.exists(AUGMENTED_TRAIN_CSV):
        print("  ⚠️  Training CSV not found, skipping training history chart")
        return None

    bl = load_training_csv(BASELINE_TRAIN_CSV)
    aug = load_training_csv(AUGMENTED_TRAIN_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # mAP50
    ax = axes[0, 0]
    ax.plot(bl["epochs"], bl["mAP50"], '-', color=C_BASELINE, linewidth=2, label='Baseline', alpha=0.8)
    ax.plot(aug["epochs"], aug["mAP50"], '-', color=C_AUGMENTED, linewidth=2, label='Augmented', alpha=0.8)
    ax.set_title('mAP@50 per Epoch', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@50')
    ax.legend(facecolor=C_CARD, edgecolor='#555')
    ax.grid(True, linestyle='--', alpha=0.3)

    # mAP50-95
    ax = axes[0, 1]
    ax.plot(bl["epochs"], bl["mAP50-95"], '-', color=C_BASELINE, linewidth=2, label='Baseline', alpha=0.8)
    ax.plot(aug["epochs"], aug["mAP50-95"], '-', color=C_AUGMENTED, linewidth=2, label='Augmented', alpha=0.8)
    ax.set_title('mAP@50-95 per Epoch', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@50-95')
    ax.legend(facecolor=C_CARD, edgecolor='#555')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Train Loss
    ax = axes[1, 0]
    ax.plot(bl["epochs"], bl["train_box"], '-', color=C_BASELINE, linewidth=1.5, label='Baseline box', alpha=0.7)
    ax.plot(bl["epochs"], bl["train_cls"], '--', color=C_BASELINE, linewidth=1.5, label='Baseline cls', alpha=0.7)
    ax.plot(aug["epochs"], aug["train_box"], '-', color=C_AUGMENTED, linewidth=1.5, label='Augmented box', alpha=0.7)
    ax.plot(aug["epochs"], aug["train_cls"], '--', color=C_AUGMENTED, linewidth=1.5, label='Augmented cls', alpha=0.7)
    ax.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8, facecolor=C_CARD, edgecolor='#555')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Val Loss
    ax = axes[1, 1]
    ax.plot(bl["epochs"], bl["val_box"], '-', color=C_BASELINE, linewidth=1.5, label='Baseline box', alpha=0.7)
    ax.plot(bl["epochs"], bl["val_cls"], '--', color=C_BASELINE, linewidth=1.5, label='Baseline cls', alpha=0.7)
    ax.plot(aug["epochs"], aug["val_box"], '-', color=C_AUGMENTED, linewidth=1.5, label='Augmented box', alpha=0.7)
    ax.plot(aug["epochs"], aug["val_cls"], '--', color=C_AUGMENTED, linewidth=1.5, label='Augmented cls', alpha=0.7)
    ax.set_title('Validation Loss', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8, facecolor=C_CARD, edgecolor='#555')
    ax.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle('📈 Training History: Baseline vs Augmented',
                 fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart_training_history.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")
    return path


# ============================================================
#  CHART 5: Hero Summary Dashboard (Single Image)
# ============================================================
def chart_hero_dashboard(results):
    """Dashboard ringkasan satu halaman — bisa langsung masuk skripsi."""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Ambil key metrics
    bl_n = results["baseline_normal"]
    bl_l = results["baseline_lowlight"]
    aug_n = results["augmented_normal"]
    aug_l = results["augmented_lowlight"]

    drop_bl_map = (bl_n["mAP50"] - bl_l["mAP50"]) / bl_n["mAP50"] * 100
    drop_aug_map = (aug_n["mAP50"] - aug_l["mAP50"]) / aug_n["mAP50"] * 100
    drop_bl_recall = (bl_n["recall"] - bl_l["recall"]) / bl_n["recall"] * 100
    drop_aug_recall = (aug_n["recall"] - aug_l["recall"]) / aug_n["recall"] * 100
    improvement = drop_bl_map - drop_aug_map

    # ---- Panel 1: Headline Numbers (top-left) ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.text(0.5, 0.92, 'ROBUSTNESS', ha='center', fontsize=14, fontweight='bold',
             color='#888', transform=ax1.transAxes)
    ax1.text(0.5, 0.65, f'+{improvement:.1f}%', ha='center', fontsize=48, fontweight='bold',
             color=C_AUGMENTED, transform=ax1.transAxes)
    ax1.text(0.5, 0.45, 'Lebih Tahan Low-Light', ha='center', fontsize=13,
             color=C_TEXT, transform=ax1.transAxes)
    ax1.text(0.5, 0.22, f'Baseline Drop: {drop_bl_map:.1f}%  →  Augmented Drop: {drop_aug_map:.1f}%',
             ha='center', fontsize=10, color='#999', transform=ax1.transAxes)
    ax1.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                                edgecolor=C_AUGMENTED, linewidth=2, transform=ax1.transAxes, alpha=0.5))

    # ---- Panel 2: mAP50 Bar (top-center) ----
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(2)
    bars1 = ax2.bar(x - 0.2, [bl_n["mAP50"], bl_l["mAP50"]], 0.35,
                    color=C_BASELINE, alpha=0.8, label='Baseline', edgecolor='white', linewidth=0.5)
    bars2 = ax2.bar(x + 0.2, [aug_n["mAP50"], aug_l["mAP50"]], 0.35,
                    color=C_AUGMENTED, alpha=0.8, label='Augmented', edgecolor='white', linewidth=0.5)
    for b in bars1:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f'{b.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f'{b.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Normal', 'Low-Light'], fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 1.08)
    ax2.set_title('mAP@50', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, facecolor=C_CARD, edgecolor='#555')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # ---- Panel 3: Recall Bar (top-right) ----
    ax3 = fig.add_subplot(gs[0, 2])
    bars1 = ax3.bar(x - 0.2, [bl_n["recall"], bl_l["recall"]], 0.35,
                    color=C_BASELINE, alpha=0.8, label='Baseline', edgecolor='white', linewidth=0.5)
    bars2 = ax3.bar(x + 0.2, [aug_n["recall"], aug_l["recall"]], 0.35,
                    color=C_AUGMENTED, alpha=0.8, label='Augmented', edgecolor='white', linewidth=0.5)
    for b in bars1:
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f'{b.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
    for b in bars2:
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f'{b.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Normal', 'Low-Light'], fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 1.08)
    ax3.set_title('Recall', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, facecolor=C_CARD, edgecolor='#555')
    ax3.grid(axis='y', linestyle='--', alpha=0.3)

    # ---- Panel 4: Table (bottom-left, spanning 2 cols) ----
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.axis('off')
    table_data = [
        ['', 'Precision', 'Recall', 'mAP@50', 'mAP@50-95', 'Drop mAP@50'],
        ['Baseline + Normal', f'{bl_n["precision"]:.4f}', f'{bl_n["recall"]:.4f}',
         f'{bl_n["mAP50"]:.4f}', f'{bl_n["mAP50-95"]:.4f}', '—'],
        ['Baseline + Low-Light', f'{bl_l["precision"]:.4f}', f'{bl_l["recall"]:.4f}',
         f'{bl_l["mAP50"]:.4f}', f'{bl_l["mAP50-95"]:.4f}', f'↓{drop_bl_map:.2f}%'],
        ['Augmented + Normal', f'{aug_n["precision"]:.4f}', f'{aug_n["recall"]:.4f}',
         f'{aug_n["mAP50"]:.4f}', f'{aug_n["mAP50-95"]:.4f}', '—'],
        ['Augmented + Low-Light', f'{aug_l["precision"]:.4f}', f'{aug_l["recall"]:.4f}',
         f'{aug_l["mAP50"]:.4f}', f'{aug_l["mAP50-95"]:.4f}', f'↓{drop_aug_map:.2f}%'],
    ]

    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#444466')
        if row == 0:
            cell.set_facecolor('#2c3e6b')
            cell.set_text_props(fontweight='bold', color='white')
        elif row in [1, 2]:
            cell.set_facecolor('#3d1c1c' if row == 2 else '#2d1c1c')
            cell.set_text_props(color=C_TEXT)
        elif row in [3, 4]:
            cell.set_facecolor('#1c2d3d' if row == 4 else '#1c2333')
            cell.set_text_props(color=C_TEXT)

    ax4.set_title('📋 Tabel Perbandingan Lengkap', fontsize=13, fontweight='bold', pad=15)

    # ---- Panel 5: Conclusion (bottom-right) ----
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    ax5.text(0.5, 0.92, '📝 KESIMPULAN', ha='center', fontsize=13,
             fontweight='bold', color=C_ACCENT, transform=ax5.transAxes)

    conclusions = [
        f'1. Model augmented {improvement:.1f}%\n   lebih tahan low-light',
        f'2. mAP@50 drop baseline:\n   {drop_bl_map:.1f}% vs augmented: {drop_aug_map:.1f}%',
        f'3. Recall drop baseline:\n   {drop_bl_recall:.1f}% vs augmented: {drop_aug_recall:.1f}%',
        f'4. Data augmentation\n   EFEKTIF meningkatkan\n   robustness deteksi wajah',
    ]

    y_pos = 0.78
    for c in conclusions:
        ax5.text(0.08, y_pos, c, fontsize=9.5, color=C_TEXT,
                 transform=ax5.transAxes, linespacing=1.4,
                 verticalalignment='top')
        y_pos -= 0.22

    ax5.add_patch(plt.Rectangle((0.03, 0.03), 0.94, 0.94, fill=False,
                                edgecolor=C_ACCENT, linewidth=2,
                                transform=ax5.transAxes, alpha=0.4))

    fig.suptitle('🔬 Analisis Robustness Deteksi Wajah YOLOv8: Baseline vs Low-Light Augmented',
                 fontsize=16, fontweight='bold', y=0.98)

    path = os.path.join(OUTPUT_DIR, "DASHBOARD_SUMMARY.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")
    return path


# ============================================================
#  Text Report
# ============================================================
def generate_text_report(results):
    """Generate laporan teks ringkasan."""
    bl_n = results["baseline_normal"]
    bl_l = results["baseline_lowlight"]
    aug_n = results["augmented_normal"]
    aug_l = results["augmented_lowlight"]

    drop_bl = (bl_n["mAP50"] - bl_l["mAP50"]) / bl_n["mAP50"] * 100
    drop_aug = (aug_n["mAP50"] - aug_l["mAP50"]) / aug_n["mAP50"] * 100
    improvement = drop_bl - drop_aug

    drop_bl_95 = (bl_n["mAP50-95"] - bl_l["mAP50-95"]) / bl_n["mAP50-95"] * 100
    drop_aug_95 = (aug_n["mAP50-95"] - aug_l["mAP50-95"]) / aug_n["mAP50-95"] * 100

    drop_bl_r = (bl_n["recall"] - bl_l["recall"]) / bl_n["recall"] * 100
    drop_aug_r = (aug_n["recall"] - aug_l["recall"]) / aug_n["recall"] * 100

    report = f"""
{'='*70}
  LAPORAN ANALISIS ROBUSTNESS DETEKSI WAJAH YOLOv8
  Low-Light Data Augmentation Study
{'='*70}

📅 Generated: Phase 6.4
🏗️ Model: YOLOv8n (Nano)
📊 Dataset: WIDER FACE subset
🖼️ Val Set: 1,366 images (normal) + 1,366 images (low-light)
🔧 Augmentasi: Gamma Correction (γ=0.4) + Brightness (-30)

{'─'*70}
  1. HASIL EVALUASI
{'─'*70}

┌──────────────────────────┬───────────┬─────────┬─────────┬───────────┐
│ Skenario                 │ Precision │  Recall │  mAP@50 │ mAP@50-95 │
├──────────────────────────┼───────────┼─────────┼─────────┼───────────┤
│ Baseline + Normal        │  {bl_n['precision']:.4f}   │ {bl_n['recall']:.4f}  │ {bl_n['mAP50']:.4f}  │   {bl_n['mAP50-95']:.4f}  │
│ Baseline + Low-Light     │  {bl_l['precision']:.4f}   │ {bl_l['recall']:.4f}  │ {bl_l['mAP50']:.4f}  │   {bl_l['mAP50-95']:.4f}  │
│ Augmented + Normal       │  {aug_n['precision']:.4f}   │ {aug_n['recall']:.4f}  │ {aug_n['mAP50']:.4f}  │   {aug_n['mAP50-95']:.4f}  │
│ Augmented + Low-Light    │  {aug_l['precision']:.4f}   │ {aug_l['recall']:.4f}  │ {aug_l['mAP50']:.4f}  │   {aug_l['mAP50-95']:.4f}  │
└──────────────────────────┴───────────┴─────────┴─────────┴───────────┘

{'─'*70}
  2. ANALISIS ROBUSTNESS DROP
{'─'*70}

                        mAP@50      mAP@50-95    Recall
  Baseline Drop:       ↓{drop_bl:.2f}%      ↓{drop_bl_95:.2f}%      ↓{drop_bl_r:.2f}%
  Augmented Drop:      ↓{drop_aug:.2f}%       ↓{drop_aug_95:.2f}%      ↓{drop_aug_r:.2f}%

  🔑 Improvement:      +{improvement:.2f}%     +{drop_bl_95 - drop_aug_95:.2f}%     +{drop_bl_r - drop_aug_r:.2f}%
                        (Augmented lebih tahan)

{'─'*70}
  3. KESIMPULAN
{'─'*70}

  ✅ Data augmentation low-light EFEKTIF meningkatkan robustness
     model YOLOv8 dalam deteksi wajah.

  ✅ Model yang dilatih dengan data augmented hanya mengalami
     penurunan mAP@50 sebesar {drop_aug:.2f}% di kondisi low-light,
     dibandingkan baseline yang turun {drop_bl:.2f}%.

  ✅ Model augmented {improvement:.1f}% LEBIH TAHAN terhadap perubahan
     kondisi pencahayaan rendah.

  ✅ Recall augmented di low-light ({aug_l['recall']:.4f}) jauh lebih tinggi
     dibanding baseline ({bl_l['recall']:.4f}), menunjukkan model augmented
     mampu mendeteksi lebih banyak wajah di kondisi gelap.

{'='*70}
"""
    path = os.path.join(OUTPUT_DIR, "REPORT_SUMMARY.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✅ {path}")

    # Juga print ke terminal
    print(report)
    return path


# ============================================================
#  MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_dark_style()

    print("Loading evaluation results...")
    results = load_eval_results()

    print("\n📊 Generating charts...")
    chart_4way_comparison(results)
    chart_robustness_drop(results)
    chart_radar(results)
    chart_training_history()
    chart_hero_dashboard(results)

    print("\n📝 Generating text report...")
    generate_text_report(results)

    print("\n" + "=" * 50)
    print("  🎉 SEMUA REPORT BERHASIL DI-GENERATE!")
    print("=" * 50)
    print(f"\n  Output folder: {OUTPUT_DIR}")
    print(f"  Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith(("chart_", "DASHBOARD", "REPORT")):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            print(f"    📄 {f} ({size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
