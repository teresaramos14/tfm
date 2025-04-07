import cv2
import numpy as np
import os
import csv  

""" He creado un archivo CSV que contiene los nombres de las imágenes y los números de las celdas de interés. 
El programa lee este archivo y procesa únicamente las imágenes especificadas. 
En cada imagen, se detecta y recorta automáticamente la caja de Petri, ya que el fondo podía interferir en la segmentación.
A continuación, se identifican y extraen las divisiones internas de la placa (celdas), las cuales se enumeran y guardan individualmente. 
Por último, se segmentan los callos presentes en la celda indicada en el CSV. Todos los resultados —tanto las celdas como los callos— 
se guardan de forma organizada en carpetas correspondientes a cada imagen. """

# Función para cargar y convertir la imagen
def load_image(image_path, convert_rgb=True):
    image = cv2.imread(image_path)
    if convert_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Función para segmentar la placa de Petri
def segment_petri_dish(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0) # Si cambio a 15, 15 sale p79,p81 y p165
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
        if w > 150 and h > 150:  # Filtrar celdas demasiado pequeñas
            sorted_cells.append((x, y, w, h, c))

    sorted_cells.sort(key=lambda x: (x[1], x[0]))  # Ordenar por fila y columna
    
    # Crear carpeta para cada imagen
    image_folder = os.path.join(petri_output_folder, image_name)
    os.makedirs(image_folder, exist_ok=True)  # Crear carpeta si no existe
    
    # Guardar las celdas en la carpeta correspondiente
    for i, (x, y, w, h, _) in enumerate(sorted_cells, 1):
        cell = image[y:y+h, x:x+w]
        cell_filename = os.path.join(image_folder, f"celda_{i}.png")
        cv2.imwrite(cell_filename, cell)
        print(f"Celda {i} guardada en: {cell_filename}")
        cv2.imshow(f"Cell {i}", cell)
        cv2.waitKey(5)

# Función para extraer los callos de una celda
def extract_callos(cell_image, output_folder, image_name, cell_number):
    try:
        gray = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
        # Umbralización adaptativa
        _, thresh = cv2.threshold(gray, 100, 280, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        callos = []

        # Filtrar y almacenar los contornos de los callos
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 30 and h > 30:  
                callos.append((x, y, w, h))

        # Crear subcarpeta para callos dentro de la carpeta de la imagen
        callos_folder = os.path.join(output_folder, image_name)
        os.makedirs(callos_folder, exist_ok=True)

        for i, (x, y, w, h) in enumerate(callos, 1):
            callo = cell_image[y:y+h, x:x+w]
            
            # Incluir el número de la celda en el nombre del archivo para evitar sobreescritura
            callo_filename = os.path.join(callos_folder, f"celda_{cell_number}_callo_{i}.png")
            
            cv2.imwrite(callo_filename, cv2.cvtColor(callo, cv2.COLOR_RGB2BGR))  
            print(f"Callo {i} de la celda {cell_number} guardado en: {callo_filename}")
            cv2.imshow(f"Callo {i}", callo)
            cv2.waitKey(5)
    
    except Exception as e:
        print(f"Error al procesar los callos de la celda {cell_number} en la imagen {image_name}: {e}")



# Función para procesar las imágenes desde el archivo CSV
def process_from_csv(csv_file, petri_output_folder, callos_output_folder):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            image_name = row['image_name']
            cell_number = row['cell_number']
            image_path = f"C:/Users/tere1/OneDrive/Escritorio/TFM/imagenes_prueba/{image_name}.png"
            print(f"Procesando la imagen: {image_name} y la celda: {cell_number}")
            
            # Llamar al proceso de la imagen y la celda
            process_image(image_path, petri_output_folder, callos_output_folder, cell_number, image_name)

# Función principal para todo el proceso
def process_image(image_path, petri_output_folder, callos_output_folder, cell_number, image_name):
    try:
        # Cargar la imagen original
        image = load_image(image_path)
        if image is None:
            print(f"No se pudo cargar la imagen: {image_path}")
            return

        # Recortar la placa de Petri
        cropped_image, _, _ = segment_petri_dish(image)
        if cropped_image is None:
            print(f"No se detectó la placa de Petri en la imagen: {image_path}")
            return
        else:
            print("Placa de Petri recortada correctamente.")
        cv2.imshow("Placa Recortada", cropped_image)
        cv2.waitKey(3)

        # Preprocesar la imagen para detectar la cuadrícula
        processed_image = preprocess_image(cropped_image)

        # Detectar las líneas y la cuadrícula
        vertical_lines, horizontal_lines = detect_lines(processed_image)
        grid_points = detect_grid_points(vertical_lines, horizontal_lines)

        # Extraer las celdas y guardarlas en la carpeta correspondiente
        extract_cells(cropped_image, grid_points, image_name, petri_output_folder)

        # Procesar la celda específica 
        cell_image = load_image(os.path.join(petri_output_folder, image_name, f"celda_{cell_number}.png"), convert_rgb=False)
        extract_callos(cell_image, callos_output_folder, image_name, cell_number)
    
    except Exception as e:
        print(f"Error al procesar la imagen {image_name}: {e}")


# Llamada a la función con el archivo CSV
csv_file = r"C:\Users\tere1\OneDrive\Escritorio\TFM\imagenes.csv"  
petri_output_folder = r"C:/Users/tere1/OneDrive/Escritorio/TFM/celdas_extraidas"
callos_output_folder = r"C:/Users/tere1/OneDrive/Escritorio/TFM/callos_extraidos"

process_from_csv(csv_file, petri_output_folder, callos_output_folder)
