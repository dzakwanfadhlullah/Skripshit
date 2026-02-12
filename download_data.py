import os
import shutil
import kagglehub

def download_and_move(dataset_name, target_folder):
    """
    Mendownload dataset dari Kaggle dan memindahkannya ke project folder.
    """
    print(f"\n--- Mendownload {dataset_name} ---")
    
    # Download via kagglehub (ke folder temp)
    tmp_path = kagglehub.dataset_download(dataset_name)
    
    # Pastikan target folder ada
    os.makedirs(target_folder, exist_ok=True)
    
    print(f"Selesai download ke: {tmp_path}")
    print(f"Memindahkan data ke: {target_folder}...")
    
    # Memindahkan isi folder temp ke target folder
    for item in os.listdir(tmp_path):
        s = os.path.join(tmp_path, item)
        d = os.path.join(target_folder, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
            
    print(f"Sukses! Data {dataset_name} kini ada di {target_folder}")

if __name__ == "__main__":
    # Path project kita (sesuaikan jika di Colab)
    # Di Colab harus diganti ke path Drive yang kita cari tadi
    PROJECT_ROOT = "." 
    
    # 1. Dataset Pelatihan Utama (Normal lighting untuk di-augmentasi)
    # dataset_train_path = os.path.join(PROJECT_ROOT, "datasets/face_detection_normal")
    # download_and_move("fareselmenshawii/face-detection-dataset", dataset_train_path)
    
    # 2. Dataset Pengujian Robustness (Real low light)
    # dataset_test_path = os.path.join(PROJECT_ROOT, "datasets/dark_face_real")
    # download_and_move("soumikrakshit/dark-face-dataset", dataset_test_path)
    
    print("\nScript siap digunakan. Silakan uncomment baris download yang dibutuhkan di Google Colab.")
