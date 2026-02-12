import cv2
import numpy as np
import os

def adjust_gamma(image, gamma=1.0):
    """
    Mengaplikasikan Gamma Correction pada gambar.
    gamma < 1.0 akan menggelapkan gambar.
    gamma > 1.0 akan mencerahkan gambar.
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_brightness(image, value=-50):
    """
    Mengubah kecerahan gambar secara linear.
    value negatif untuk menggelapkan.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Tambahkan brightness dengan clipping
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def augment_to_low_light(img_path, output_path, gamma=0.4, brightness=-30):
    """
    Fungsi gabungan untuk simulasi low-light (pencahayaan rendah).
    """
    img = cv2.imread(img_path)
    if img is None:
        return False
    
    # 1. Gelapkan dengan Gamma (Non-linear)
    img_low = adjust_gamma(img, gamma=gamma)
    
    # 2. Kurangi Brightness (Linear)
    img_low = adjust_brightness(img_low, value=brightness)
    
    cv2.imwrite(output_path, img_low)
    return True

if __name__ == "__main__":
    # Contoh test satu gambar
    # augment_to_low_light("test.jpg", "test_low.jpg")
    pass
