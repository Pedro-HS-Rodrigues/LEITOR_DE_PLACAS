from detect_plate import detect_license_plate
from ocr import extract_text_from_plate

import cv2
import numpy as np

def main():
    image_path = "data/raw/image1.jpg"
    plate_img, plate_text = detect_license_plate(image_path)

    if plate_img is not None and isinstance(plate_img, (np.ndarray, )):
        print("Texto detectado:", plate_text)
        cv2.imwrite("data/processed/detected_plate.jpg", plate_img)
    else:
        print("Nenhuma placa válida encontrada.")

if __name__ == "__main__":
    main()
