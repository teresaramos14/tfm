import cv2
import numpy as np
import os
import csv  

# Función para cargar y convertir la imagen
def load_image(image_path, convert_rgb=True):
    image = cv2.imread(image_path)
    if convert_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Función para segmentar la placa de Petri
def segment_petri_dish(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0) 
    edges = cv2.Canny(blurred, 30, 100)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    petri_contour = contours[0] if contours else None

    if petri_contour is not None:
        epsilon = 0.02 * cv2.arcLength(petri_contour, True)
        approx = cv2.approxPolyDP(petri_contour, epsilon, True)
        if len(approx) == 4:
            approx = approx.reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(approx)
            return image[y:y+h, x:x+w], approx, (x, y, w, h)
    
    return None, None, None


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

    sorted_cells = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 150 and h > 150:  
            sorted_cells.append((x, y, w, h, c))

    sorted_cells.sort(key=lambda x: (x[1], x[0]))  
    
    image_folder = os.path.join(petri_output_folder, image_name)
    os.makedirs(image_folder, exist_ok=True)  
    
    for i, (x, y, w, h, _) in enumerate(sorted_cells, 1):
        cell = image[y:y+h, x:x+w]
        cell_filename = os.path.join(image_folder, f"celda_{i}.png")
        cv2.imwrite(cell_filename, cell)
        print(f"Celda {i} guardada en: {cell_filename}")
        cv2.imshow(f"Cell {i}", cell)
        cv2.waitKey(5)

# Función para extraer los callos utilizando la detección de color
def extract_callos_by_color(cell_image):
    hsv = cv2.cvtColor(cell_image, cv2.COLOR_RGB2HSV)

    # Definir el rango de color para los callos oscuros
    lower_callo_oscuro = np.array([0, 0, 50])  # Límite inferior para callos oscuros
    upper_callo_oscuro = np.array([180, 255, 150])  # Límite superior para callos oscuros
    mask_callo_oscuro = cv2.inRange(hsv, lower_callo_oscuro, upper_callo_oscuro)

    # Mostrar la máscara de los callos oscuros para depuración
    cv2.imshow("Máscara Callos Oscuros", mask_callo_oscuro)

    return mask_callo_oscuro

# Función para extraer y guardar los callos recortados de las celdas
def extract_callos(cell_image, output_folder, image_name, cell_number):
    try:
        # Llamar a la función de extracción de callos por color
        mask = extract_callos_by_color(cell_image)

        # Aplicar una operación morfológica para eliminar ruidos (opcional)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Kernel más grande para eliminar pequeños ruidos
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Encontrar los contornos en la máscara limpia
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        callos = []

        # Recorrer los contornos y asegurarse de que sean lo suficientemente grandes para ser considerados callos
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 30 and h > 30:  # Ajusta el tamaño mínimo según sea necesario
                callos.append((x, y, w, h))

        if not callos:
            print("No se detectaron callos en esta celda.")
            return  # Si no hay callos detectados, salimos

        # Carpeta para almacenar los callos
        callos_folder = os.path.join(output_folder, image_name)
        os.makedirs(callos_folder, exist_ok=True)

        # Guardar la máscara de callos oscuros
        mask_filename = os.path.join(callos_folder, f"mascara_callos_celda_{cell_number}.png")
        cv2.imwrite(mask_filename, mask_cleaned)
        print(f"Máscara de callos guardada en: {mask_filename}")

        # Para cada callo detectado, recortarlo y guardarlo
        for i, (x, y, w, h) in enumerate(callos, 1):
            callo = cell_image[y:y+h, x:x+w]
            callo_filename = os.path.join(callos_folder, f"celda_{cell_number}_callo_{i}.png")
            cv2.imwrite(callo_filename, cv2.cvtColor(callo, cv2.COLOR_RGB2BGR))  # Convertir de RGB a BGR para guardarlo correctamente
            print(f"Callo {i} de la celda {cell_number} guardado en: {callo_filename}")
            cv2.imshow(f"Callo {i}", callo)
            cv2.waitKey(5)  # Muestra cada callo temporalmente

    except Exception as e:
        print(f"Error al procesar los callos de la celda {cell_number} en la imagen {image_name}: {e}")


# Función para procesar las imágenes desde el archivo CSV
def process_from_csv(csv_file, petri_output_folder, callos_output_folder):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            image_name = row['image_name']
            cell_number = row['cell_number']
            image_path = f"C:/Users/tere1/OneDrive/Escritorio/TFM/250406/imagenes_prueba/{image_name}.png"
            print(f"Procesando la imagen: {image_name} y la celda: {cell_number}")
            
            process_image(image_path, petri_output_folder, callos_output_folder, cell_number, image_name)

# Función principal para todo el proceso
def process_image(image_path, petri_output_folder, callos_output_folder, cell_number, image_name):
    try:
        image = load_image(image_path)
        if image is None:
            print(f"No se pudo cargar la imagen: {image_path}")
            return

        cropped_image, _, _ = segment_petri_dish(image)
        if cropped_image is None:
            print(f"No se detectó la placa de Petri en la imagen: {image_path}")
            return
        else:
            print("Placa de Petri recortada correctamente.")
        cv2.imshow("Placa Recortada", cropped_image)
        cv2.waitKey(50)

        processed_image = preprocess_image(cropped_image)

        vertical_lines, horizontal_lines = detect_lines(processed_image)
        grid_points = detect_grid_points(vertical_lines, horizontal_lines)

        extract_cells(cropped_image, grid_points, image_name, petri_output_folder)

        cell_image = load_image(os.path.join(petri_output_folder, image_name, f"celda_{cell_number}.png"), convert_rgb=False)
        extract_callos(cell_image, callos_output_folder, image_name, cell_number)
    
    except Exception as e:
        print(f"Error al procesar la imagen {image_name}: {e}")


# Llamada a la función con el archivo CSV
csv_file = r"C:\Users\tere1\OneDrive\Escritorio\TFM\250406/imagenes.csv"  
petri_output_folder = r"C:/Users/tere1/OneDrive/Escritorio/TFM/250406/celdas_extraidas"
callos_output_folder = r"C:/Users/tere1/OneDrive/Escritorio/TFM/250406/callos_extraidos"

process_from_csv(csv_file, petri_output_folder, callos_output_folder)
