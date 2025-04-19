import cv2
import numpy as np
import os
import csv

"""Este código procesa imágenes de placas de Petri listadas en un archivo CSV, segmentando cada imagen para detectar la placa mediante técnicas de desenfoque
 y detección de bordes; luego identifica y extrae las celdas de la cuadrícula usando operaciones morfológicas,
y finalmente, en una celda específica indicada por el CSV, detecta los callos en función del color o la intensidad según el 
fondo sea claro u oscuro. Los callos detectados se recortan y guardan como imágenes independientes, manteniendo el fondo original,
 mientras que las celdas extraídas y los callos se almacenan en carpetas organizadas por nombre de imagen y variante de procesamiento."""

# Función para cargar y convertir la imagen
def load_image(image_path, convert_rgb=True): 
    image = cv2.imread(image_path) 
    if convert_rgb: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Función para segmentar la placa de Petri
def segment_petri_dish(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur_values = [(5,5), (7,7), (9,9), (15,15)]
    canny_thresholds = [(20, 80), (30, 100), (50, 150), (70, 200)]

    valid_crops = []

    for i, blur in enumerate(blur_values):
        blurred = cv2.GaussianBlur(gray, blur, 0)

        for j, (th1, th2) in enumerate(canny_thresholds):
            edges = cv2.Canny(blurred, th1, th2)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 10000:
                    continue

                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h

                if 0.7 < aspect_ratio < 1.2:
                    margin = 10  # Ajusta el margen según sea necesario

                    img_height, img_width = image.shape[:2]
                    x1 = max(x - margin, 0)
                    y1 = max(y - margin, 0)
                    x2 = min(x + w + margin, img_width)
                    y2 = min(y + h + margin, img_height)

                    cropped = image[y1:y2, x1:x2]
                    cropped_area = cropped.shape[0] * cropped.shape[1]

                    if 3000000 <= cropped_area <= 4500000:
                        label = f"blur_{i}_{blur[0]}x{blur[1]}_canny_{th1}_{th2}"
                        print(f"[{label}] Recorte válido: {cropped.shape[0]}x{cropped.shape[1]} ({cropped_area})")
                        valid_crops.append((cropped.copy(), label))
                        break  # Solo una placa por combinación de blur + canny

    return valid_crops

# Función para aplicar umbral adaptativo y operación de cierre 
def preprocess_image(image): 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 57, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Función para detectar líneas verticales y horizontales
def detect_lines(thresh_image):
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    vertical_lines = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, vertical_kernel)
    horizontal_lines = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, horizontal_kernel)

    return vertical_lines, horizontal_lines

# Función para detectar y corregir los puntos de intersección
def detect_grid_points(vertical_lines, horizontal_lines):
    grid_points = cv2.bitwise_and(vertical_lines, horizontal_lines)
 
    return cv2.dilate(grid_points, None, iterations=2)
 
# Función para extraer celdas de la cuadrícula
def extract_cells(image, grid_points, image_name, petri_output_folder):
    inverted_grid = cv2.bitwise_not(grid_points)
    contours, _ = cv2.findContours(inverted_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cells = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w > 150 and h > 150:  # Filtrar celdas pequeñas
            cells.append((x, y, w, h, c))

    # Ordenar directamente por fila (Y) y luego por columna (X)
    sorted_cells = sorted(cells, key=lambda x: (x[1], x[0]))


    # Crear carpeta para la imagen
    image_folder = os.path.join(petri_output_folder, image_name)
    os.makedirs(image_folder, exist_ok=True)

    # Guardar las celdas ordenadas
    for i, (x, y, w, h, _) in enumerate(sorted_cells, 1):
        margin = 9  # puedes ajustar este valor
        img_height, img_width = image.shape[:2]

        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, img_width)
        y2 = min(y + h + margin, img_height)

        cell = image[y1:y2, x1:x2]
        cell_filename = os.path.join(image_folder, f"celda_{i}.png")
        cv2.imwrite(cell_filename, cell)

        print(f"Celda {i} guardada en: {cell_filename}")
        cv2.imshow(f"Celda {i}", cell)
        cv2.waitKey(5)

# Función para extraer callos 
def extract_callos(cell_image_blurred, cell_image_original, output_folder, image_name, cell_number, blur_label):
    try:
        # Convertir la celda original a escala de grises
        gray = cv2.cvtColor(cell_image_original, cv2.COLOR_RGB2GRAY)
        mean_intensity = np.mean(gray)

        print(f"[INFO] Promedio de intensidad de la celda {cell_number}: {mean_intensity}")

        callos_folder = os.path.join(output_folder, image_name, blur_label)
        os.makedirs(callos_folder, exist_ok=True)

        if mean_intensity < 136:
            print("[INFO] Detección de callos oscuros.")
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            print("[INFO] La celda es clara, detectando callos por color.")
            hsv = cv2.cvtColor(cell_image_original, cv2.COLOR_RGB2HSV)
            lower_color = np.array([0, 0, 50])
            upper_color = np.array([180, 255, 150])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        callo_index = 1
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 20 and h > 20:
                # Crear máscara del contorno detectado
                mask = np.zeros(cell_image_original.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [c], -1, 255, -1)

                # Aplicar la máscara para extraer el contorno de la imagen original
                masked_callo = cv2.bitwise_and(cell_image_original, cell_image_original, mask=mask)

                # La imagen original que no está cubierta por el callo
                background = cv2.bitwise_and(cell_image_original, cell_image_original, mask=cv2.bitwise_not(mask))

                # Pegar el callo sobre el fondo original 
                result_image = cv2.add(background, masked_callo)

                # Calcular el recorte a partir de la máscara
                ys, xs = np.where(mask == 255)
                if len(xs) == 0 or len(ys) == 0:
                    continue

                margin = 5
                x1 = max(np.min(xs) - margin, 0)
                y1 = max(np.min(ys) - margin, 0)
                x2 = min(np.max(xs) + margin, cell_image_original.shape[1])
                y2 = min(np.max(ys) + margin, cell_image_original.shape[0])

                # Recorte de la imagen con fondo original
                callo_roi = result_image[y1:y2, x1:x2]

                # Guardar imagen con el fondo original
                callo_filename = os.path.join(callos_folder, f"celda_{cell_number}_callo_{callo_index}.png")
                cv2.imwrite(callo_filename, cv2.cvtColor(callo_roi, cv2.COLOR_RGB2BGR))
                print(f"Callo {callo_index} de la celda {cell_number} guardado en: {callo_filename}")
                cv2.imshow(f"Callo {callo_index}", callo_roi)
                cv2.waitKey(5)

                callo_index += 1

        if callo_index == 1:
            print("No se detectaron callos en esta celda.")

    except Exception as e:
        print(f"Error al procesar los callos de la celda {cell_number} en la imagen {image_name}: {e}")

# Función para procesar las imágenes desde el archivo CSV
def process_from_csv(csv_file, petri_output_folder, callos_output_folder):
    with open(csv_file, mode='r', encoding='utf-8') as file:
 
        reader = csv.DictReader(file, delimiter=';')
        for row in reader: 
            image_name = row['image_name']
            cell_number = row['cell_number']
            image_path = f"C:/Users/tere1/OneDrive/Escritorio/TFM/250415/imagenes/higher_than/{image_name}.png"
 
            print(f"Procesando la imagen: {image_name} y la celda: {cell_number}")

            # Llamar al proceso de la imagen y la celda
            process_image(image_path, petri_output_folder, callos_output_folder, cell_number, image_name)
 
# Función principal para todo el proceso
def process_image(image_path, petri_output_folder, callos_output_folder, cell_number, image_name):
    try:
        image = load_image(image_path)
        if image is None:
            print(f"No se pudo cargar la imagen: {image_path}")
            return

        cropped_variants = segment_petri_dish(image)

        if not cropped_variants:
            print(f"No se detectó ninguna placa válida en: {image_path}")
            return

        for cropped_image, blur_label in cropped_variants:
            print(f"[INFO] Procesando variante: {blur_label}")
            
            variant_name = f"{image_name}_{blur_label}"

            processed_image = preprocess_image(cropped_image)
            vertical_lines, horizontal_lines = detect_lines(processed_image)
            grid_points = detect_grid_points(vertical_lines, horizontal_lines)

            extract_cells(cropped_image, grid_points, variant_name, petri_output_folder)

            cell_path = os.path.join(petri_output_folder, variant_name, f"celda_{cell_number}.png")
            cell_image_blurred = load_image(cell_path, convert_rgb=False)
            cell_image_original = load_image(cell_path, convert_rgb=False)



            extract_callos(cell_image_blurred, cell_image_original, callos_output_folder, variant_name, cell_number, blur_label)


    except Exception as e:
        print(f"Error al procesar la imagen {image_name}: {e}")

  
csv_file = r"C:\Users\tere1\OneDrive\Escritorio\TFM\250415\imagenes.csv"  
petri_output_folder = r"C:/Users/tere1/OneDrive/Escritorio/TFM/250419/celdas_extraidas/higher_than"
callos_output_folder = r"C:/Users/tere1/OneDrive/Escritorio/TFM/250419/callos_extraidos/higher_than"
process_from_csv(csv_file, petri_output_folder, callos_output_folder)