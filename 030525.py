# Cargar librerías necesarias
import os
import pandas as pd
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage import color, img_as_ubyte, io
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage import io, color
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from skimage import img_as_float

# Rutas a las carpetas de imágenes
smaller_than = r"C:\Users\tere1\OneDrive\Escritorio\TFM\segmentacion de callos\results\smaller_than"
higher_than = r"C:\Users\tere1\OneDrive\Escritorio\TFM\segmentacion de callos\results\higher_than"

# Función para cargar la imagen en color
def load_image(image_path):
    return cv2.imread(image_path)

# procesar cada imagen de una carpeta
def process_images_in_folder(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        img = load_image(image_path)
        if img is None:
            continue
        data.append((filename, img))
    return data

# Función para extraer características GLCM
def extract_glcm_features(folder_path, label=None):
    data = []
    bins32 = np.linspace(0, 255, 33).astype(np.uint8)  # 32 niveles
    images_data = process_images_in_folder(folder_path)  # Obtener lista de imágenes procesadas
    
    for filename, img in images_data:
        gray = color.rgb2gray(img)
        gray = img_as_ubyte(gray)
        inds = np.digitize(gray, bins32)

        glcm = graycomatrix(
            inds,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=32,
            symmetric=False,
            normed=False
        )

        features = {
            'filename': filename,
            'contrast': np.mean(graycoprops(glcm, 'contrast')),
            'dissimilarity': np.mean(graycoprops(glcm, 'dissimilarity')),
            'homogeneity': np.mean(graycoprops(glcm, 'homogeneity')),
            'energy': np.mean(graycoprops(glcm, 'energy')),
            'correlation': np.mean(graycoprops(glcm, 'correlation')),
            'ASM': np.mean(graycoprops(glcm, 'ASM'))
        }

        if label is not None:
            features['label'] = label

        data.append(features)

    return pd.DataFrame(data)

# Función para extraer momentos de Hu
def extract_hu_moments(folder_path, label=None):
    data = []
    images_data = process_images_in_folder(folder_path)  
    
    for filename, img in images_data:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        moments = cv2.moments(binary_img)
        hu_moments = cv2.HuMoments(moments).flatten()

        features = {
            'filename': filename,
            'hu_1': hu_moments[0],
            'hu_2': hu_moments[1],
            'hu_3': hu_moments[2],
            'hu_4': hu_moments[3],
            'hu_5': hu_moments[4],
            'hu_6': hu_moments[5],
            'hu_7': hu_moments[6]
        }

        if label is not None:
            features['label'] = label

        data.append(features)

    return pd.DataFrame(data)

# Función para extraer propiedades de forma
def extract_shape_features(folder_path, label_name=None):
    data = []
    images_data = process_images_in_folder(folder_path)  # Obtener lista de imágenes procesadas
    
    for filename, image in images_data:
        # Si es RGB, convertir a escala de grises
        if image.ndim == 3:
            image = color.rgb2gray(image)

        # Umbral automático con Otsu
        thresh = threshold_otsu(image)
        binary = image > thresh

        # Etiquetar regiones conectadas
        labeled = label(binary)

        # Extraer propiedades de todas las regiones
        regions = regionprops(labeled)

        if regions:
            # Seleccionar la región más grande (la principal)
            largest_region = max(regions, key=lambda r: r.area)

            # Extraer las propiedades de la región más grande
            props = {
                'filename': filename,
                'area': largest_region.area,
                'perimeter': largest_region.perimeter,
                'eccentricity': largest_region.eccentricity,
                'extent': largest_region.extent,
                'solidity': largest_region.solidity,
                'orientation': largest_region.orientation,
                'major_axis_length': largest_region.major_axis_length,
                'minor_axis_length': largest_region.minor_axis_length
            }

            if label_name:
                props['label'] = label_name

            data.append(props)

    return pd.DataFrame(data)

# Función para vector con Filtros Gabor
def extract_gabor_features(folder_path, label=None):
    data = []
    kernels = []
    for theta in range(4):  # 0°, 45°, 90°, 135°
        theta_rad = theta / 4. * np.pi
        for sigma in (1, 3):  # Escalas
            for frequency in (0.1, 0.3):  # Frecuencias
                kernel = gabor_kernel(frequency, theta=theta_rad,
                                      sigma_x=sigma, sigma_y=sigma)
                kernels.append(kernel)

    images_data = process_images_in_folder(folder_path)  # Obtener lista de imágenes procesadas
    
    for filename, img in images_data:
        gray = color.rgb2gray(img)
        gray = img_as_float(gray)

        feats = []  # Se almacenan los vectores de características de cada imagen
        for kernel in kernels:
            filtered_real = ndi.convolve(gray, np.real(kernel), mode='wrap')
            filtered_imag = ndi.convolve(gray, np.imag(kernel), mode='wrap')
            magnitude = np.sqrt(filtered_real**2 + filtered_imag**2)
            feats.append(magnitude.mean())  # La media de la intensidad de la respuesta del filtro Gabor
            feats.append(magnitude.var())   # Dispersión de la respuesta del filtro en la imagen

        features = {'filename': filename}
        for i, val in enumerate(feats):
            features[f'gabor_feat_{i+1}'] = val

        if label is not None:
            features['label'] = label

        data.append(features)

    return pd.DataFrame(data)

# Función para realizar la Transformada de Fourier y extraer características
def extract_fourier_features(folder_path, label=None):
    data = []
    images_data = process_images_in_folder(folder_path)  # Obtener lista de imágenes procesadas
    
    for filename, img in images_data:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = img_as_float(gray)  

        # Transformada de Fourier y desplazar el cero a la mitad
        f_transform = np.fft.fft2(gray)
        f_transform_shifted = np.fft.fftshift(f_transform)  # Desplazar las frecuencias a la zona central

        # Calcula la magnitud de la transformada de Fourier
        magnitude = np.abs(f_transform_shifted)

        # Características estadísticas de la magnitud
        mean_magnitude = np.mean(magnitude)  
        std_magnitude = np.std(magnitude)    

        # Para obtener información sobre las frecuencias de bajo y alto rango
        low_freq_magnitude = np.mean(magnitude[:magnitude.shape[0]//2, :magnitude.shape[1]//2])  # Frecuencias bajas
        high_freq_magnitude = np.mean(magnitude[magnitude.shape[0]//2:, magnitude.shape[1]//2:])  # Frecuencias altas

        features = {
            'filename': filename,
            'mean_magnitude': mean_magnitude,
            'std_magnitude': std_magnitude,
            'low_freq_magnitude': low_freq_magnitude,
            'high_freq_magnitude': high_freq_magnitude
        }

        if label is not None:
            features['label'] = label

        data.append(features)

    return pd.DataFrame(data)


# Aplicar funciones a las carpetas y etiquetas proporcionadas
df_smaller_glcm = extract_glcm_features(smaller_than, label='smaller')
df_higher_glcm = extract_glcm_features(higher_than, label='higher')

df_smaller_hu = extract_hu_moments(smaller_than, label='smaller')
df_higher_hu = extract_hu_moments(higher_than, label='higher')

df_shape_smaller = extract_shape_features(smaller_than, label_name='smaller')
df_shape_higher = extract_shape_features(higher_than, label_name='higher')

df_gabor_smaller = extract_gabor_features(smaller_than, label='smaller')
df_gabor_higher = extract_gabor_features(higher_than, label='higher')

df_smaller_fourier = extract_fourier_features(smaller_than, label='smaller')
df_higher_fourier = extract_fourier_features(higher_than, label='higher')

# Mostrar cuantas imagenes procesa
print("Imágenes procesadas (smaller GLCM):", len(df_smaller_glcm))
print("Imágenes procesadas (higher GLCM):", len(df_higher_glcm))
print("Imágenes procesadas (smaller Hu):", len(df_smaller_hu))
print("Imágenes procesadas (higher Hu):", len(df_higher_hu))
print("Imágenes procesadas (smaller Shape):", len(df_shape_smaller))
print("Imágenes procesadas (higher Shape):", len(df_shape_higher))
print("Imágenes procesadas (smaller Gabor):", len(df_gabor_smaller))
print("Imágenes procesadas (higher Gabor):", len(df_gabor_higher))
print("Imágenes procesadas (smaller Fourier):", len(df_smaller_fourier))
print("Imágenes procesadas (higher Fourier):", len(df_higher_fourier))

# Mostrar los primeros registros de los DataFrames
print(df_smaller_fourier.head())
print(df_higher_fourier.head())
