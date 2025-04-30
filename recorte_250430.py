from ultralytics import YOLO
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
import herramientas_pjnl as pj

# --- Configuracion ---
nf, nc = 4, 4  # Cuadrantes
path_imag = r"C:\Users\tere1\OneDrive\Escritorio\TFM\segmentacion de callos\multi"
path_model = r"C:\Users\tere1\OneDrive\Escritorio\TFM\segmentacion de callos\models"
path_results = r"C:\Users\tere1\OneDrive\Escritorio\TFM\segmentacion de callos\results"
path_csv = os.path.join(path_results, "imagenes_a_recortar.csv")

# --- Leer CSV ---
df_targets = pd.read_csv(path_csv, delimiter=';')  # columnas: imagen,cuadrante

# --- Cargar modelo ---
model = YOLO(os.path.join(path_model, 'train_yolov8m_epo100_img_800_dataset_18_5_3.pt'))

# --- Procesamiento por imagen ---
for idx, row in df_targets.iterrows():
    filename = row['imagen']
    cuadrante_objetivo = int(row['cuadrante']) - 1  # Convertir a 0-based index

    path_img = os.path.join(path_imag, filename)
    results = model(path_img, show_labels=False, conf=0.5, iou=0.5)

    # Leer y preparar imagen
    imgBGR = cv2.imread(path_img)
    imgPETRI_BGR, pto_origen, area_placa = pj.petriExtraction(imgBGR)

    if area_placa < 3700000:
        print(f"Area placa baja: {filename}, omitida")
        continue

    imgPETRI_RGB = cv2.cvtColor(imgPETRI_BGR, cv2.COLOR_BGR2RGB)
    imgPETRIHSV = cv2.cvtColor(imgPETRI_BGR, cv2.COLOR_BGR2HSV)

    lower_green = np.array([50, 20, 10])
    upper_green = np.array([255, 100, 90])
    mask = cv2.inRange(imgPETRIHSV, lower_green, upper_green)

    CUAD = pj.calculaCuadrantes(mask)
    offset = 20
    for i in range(len(CUAD[0])):
        CUAD[0][i] = (CUAD[0][i][0], CUAD[0][i][1] + offset)

    xyxy = results[0].boxes.xyxy.cpu().numpy()
    NCallos = len(xyxy)

    output_folder = os.path.join(path_results, "callos_recortados", os.path.splitext(filename)[0])
    os.makedirs(output_folder, exist_ok=True)

    for i in range(NCallos):
        x0 = int(xyxy[i][0]) - pto_origen[0]
        y0 = int(xyxy[i][1]) - pto_origen[1]
        x1 = int(xyxy[i][2]) - pto_origen[0]
        y1 = int(xyxy[i][3]) - pto_origen[1]

        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            continue

        centro = [int((x0 + x1) / 2), int((y0 + y1) / 2)]
        cuad = pj.numeroCUAD2(centro, CUAD)

        if cuad == cuadrante_objetivo:
            callo_crop = imgPETRI_RGB[y0:y1, x0:x1]
            callo_filename = f"callo_{i+1}.png"
            cv2.imwrite(os.path.join(output_folder, callo_filename), cv2.cvtColor(callo_crop, cv2.COLOR_RGB2BGR))
            print(f"Guardado: {callo_filename} en {output_folder}")