import cv2
import pytesseract

def detect_license_plate(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Realce de contraste + limiar
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    data = pytesseract.image_to_data(thresh, config='--psm 6', output_type=pytesseract.Output.DICT)

    n_boxes = len(data['level'])
    image_copy = image.copy()

    for i in range(n_boxes):
        text = data['text'][i].strip()
        if len(text) >= 5 and any(c.isdigit() for c in text):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            plate_img = image[y:y+h, x:x+w]

            # # DEBUG: mostrar o que foi detectado
            # cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow("Possível Placa", image_copy)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            return plate_img, text

    return None, None
