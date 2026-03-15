import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from PIL import Image
import csv
import warnings

# Silenciar las advertencias molestas de PIL sobre los fondos transparentes de los PNG
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")

# Importaciones específicas de PyTorch y timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm

# ==========================================
# CONFIGURACIÓN DINÁMICA DE HARDWARE (CPU vs GPU)
# ==========================================
# Detecta si tu Quadro T2000 está disponible a través de CUDA en WSL2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
if torch.cuda.is_available():
    BATCH_SIZE = 16  # Reducimos el lote a 16 imágenes por paso para no saturar los 4GB de VRAM
    print(f"✅ GPU Detectada: {torch.cuda.get_device_name(0)}")
    print(f"✅ BATCH_SIZE ajustado a {BATCH_SIZE}.")
else:
    print(f"💻 Entrenando en CPU. BATCH_SIZE ajustado a {BATCH_SIZE}.")


# ==========================================
# 1. CONSTANTES Y PARÁMETROS GLOBALES
# ==========================================
# --- RUTAS Y ARCHIVOS ---
DATASET_PATH = Path("pokemon_datasets") / "pokemon150" # Carpeta raíz de tus imágenes
BACKUP_MODEL_PATH = 'backup_pokedex_model.pth' # Archivo temporal de seguridad durante el entrenamiento
FINAL_MODEL_PATH = 'pokedex150_resnet50v2_model.pth' # Archivo final del modelo exportado
LOG_PATH = 'historial_entrenamiento.csv' # Registro de métricas época por época
DICCIONARIO_PATH = 'clases_pokedex.json' # Archivo que mapea el índice (ej. 25) con el nombre (ej. Pikachu)
TEST_IMAGE_PATH = 'pikachu_test.png' # Imagen externa para probar la inferencia final
FINAL_VISUALIZATION_PATH = 'prediccion_final_pokedex.png' # Donde se guarda la gráfica con la predicción

# --- HIPERPARÁMETROS DE ENTRENAMIENTO ---
MODEL_NAME = 'resnetv2_50' # Arquitectura a usar desde la librería timm. Excelente balance precisión/peso.
TOTAL_EPOCAS = 20 # Número máximo de veces que la red verá el dataset completo.
LEARNING_RATE = 0.001 # Tasa de aprendizaje: qué tan grandes son los "pasos" que da el optimizador para corregir errores.
EARLY_STOPPING_PATIENCE = 3 # Si la precisión no mejora después de 3 épocas seguidas, detenemos el entrenamiento.

# --- PROCESAMIENTO DE DATOS Y SPLITS ---
IMG_SIZE = 224 # Tamaño estándar requerido por las redes ResNet (224x224 píxeles).
VAL_TEST_SPLIT_RATIO = 0.2  # Del 100% de datos, separamos 20% (10% validación para evaluar, 10% test para probar al final).
RANDOM_SEED = 123 # Semilla estática para que la división aleatoria sea siempre la misma en cada ejecución.
NUM_WORKERS = 4 # Cantidad de hilos del procesador (CPU) dedicados EXCLUSIVAMENTE a cargar imágenes desde el disco duro.

# --- DATA AUGMENTATION Y NORMALIZACIÓN ---
# Estos valores (Mean y Std) son los promedios de color oficiales del dataset ImageNet. 
# Son obligatorios para que los modelos preentrenados de timm "vean" los colores correctamente.
NORMALIZE_MEAN = [0.485, 0.456, 0.406] 
NORMALIZE_STD = [0.229, 0.224, 0.225]
FLIP_PROBABILITY = 0.5 # 50% de probabilidad de que una imagen se voltee en modo espejo.
ROTATION_DEGREES = 36 # Rotación máxima aleatoria de la imagen (de -36 a +36 grados).
AFFINE_DEGREES = 0 # Rotación adicional para la transformación afín (lo dejamos en 0 para no sobrerotar).
AFFINE_SCALE = (0.8, 1.2) # Zoom aleatorio. 0.8 acerca la imagen un 20%, 1.2 la aleja un 20%.

# --- CONFIGURACIÓN VISUAL (Gráficas) ---
CM_FIGSIZE = (20, 18) # Tamaño en pulgadas de la Matriz de Confusión.
CM_TITLE_FONTSIZE = 16 # Tamaño del texto del título de la matriz.
CM_LABEL_FONTSIZE = 12 # Tamaño del texto de los ejes X e Y de la matriz.
PRED_FIGSIZE = (6, 6) # Tamaño del lienzo para mostrar la predicción de la imagen final.
PRED_TITLE_FONTSIZE = 14 # Tamaño de la fuente del resultado de la Pokédex.


print(f"Carpeta principal encontrada en: {DATASET_PATH}")

# Definimos las tuberías de transformación (Data Augmentation)
val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), # Convierte la imagen a una matriz matemática de PyTorch (Tensor) y escala píxeles de 0-255 a 0-1.
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=FLIP_PROBABILITY),
    transforms.RandomRotation(degrees=ROTATION_DEGREES), 
    transforms.RandomAffine(degrees=AFFINE_DEGREES, scale=AFFINE_SCALE), 
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# ==========================================
# BLOQUE PRINCIPAL (Necesario para multiprocesamiento en Windows/WSL)
# ==========================================
if __name__ == '__main__':
    # ==========================================
    # 2. CARGA Y DIVISIÓN DE DATOS
    # ==========================================
    print("\n[1/6] Cargando imágenes para entrenamiento, validación y test...")

    # ImageFolder lee automáticamente los nombres de las subcarpetas y las convierte en clases/etiquetas
    full_dataset_train = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
    full_dataset_val = datasets.ImageFolder(DATASET_PATH, transform=val_test_transforms)

    class_names = full_dataset_train.classes
    with open(DICCIONARIO_PATH, 'w') as f:
        json.dump({i: nombre for i, nombre in enumerate(class_names)}, f)

    num_classes = len(class_names)

    # Creamos una lista desordenada de índices (0 al total de imágenes) para mezclarlas antes de dividirlas
    total_size = len(full_dataset_train)
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(RANDOM_SEED)).tolist()

    # Matemáticas para la división: Calculamos dónde cortar la lista de índices
    val_split = int(VAL_TEST_SPLIT_RATIO * total_size)
    train_indices = indices[val_split:] # 80% para entrenar
    val_test_indices = indices[:val_split] # 20% restante

    val_size = len(val_test_indices) // 2
    val_indices = val_test_indices[:val_size] # Mitad del 20% (10%) para Validación
    test_indices = val_test_indices[val_size:] # Mitad del 20% (10%) para Testeo final

    # Asignamos los sub-conjuntos usando Subset
    train_dataset = Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_val, val_indices)
    test_dataset = Subset(full_dataset_val, test_indices)

    # DataLoader es el "motor" que alimenta las imágenes a la GPU en lotes (Batches)
    # pin_memory=True acelera la transferencia de datos entre la memoria RAM y la memoria de la tarjeta gráfica
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


    # ==========================================
    # 3 Y 4. LÓGICA DE MODELO (CREAR O RETOMAR)
    # ==========================================
    print("\n[2/6] Verificando estado del modelo...")

    # Descarga la arquitectura y reemplaza automáticamente la última capa por una nueva de `num_classes` salidas
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)

    # Transfer Learning: Congelamos el "cuerpo" de la red para que no olvide cómo detectar formas básicas
    for param in model.parameters():
        param.requires_grad = False
    
    # Descongelamos ÚNICAMENTE la "cabeza" (la nueva capa final) para que aprenda a clasificar nuestros Pokémon
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    model = model.to(device) # Enviamos el modelo a la tarjeta gráfica

    # Definimos cómo se calcula el error (CrossEntropyLoss es el estándar para clasificación multiclase)
    criterion = nn.CrossEntropyLoss() 
    # El optimizador (Adam) actualizará matemáticamente los pesos solo de las capas descongeladas
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    epoca_inicial = 0
    best_val_acc = 0.0
    patience_counter = 0 

    # Lógica para retomar entrenamientos interrumpidos o cargar el modelo final si ya existe
    if os.path.exists(FINAL_MODEL_PATH):
        print("¡Modelo final encontrado! Cargando la Pokédex entrenada...")
        model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device, weights_only=True))
        epoca_inicial = TOTAL_EPOCAS 
    else:
        print(f"Construyendo la arquitectura de la Pokédex (timm {MODEL_NAME})...")
        if os.path.exists(BACKUP_MODEL_PATH):
            print(f"-> Retomando modelo existente desde el backup: {BACKUP_MODEL_PATH}")
            checkpoint = torch.load(BACKUP_MODEL_PATH, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict']) # Carga los "pesos" matemáticos
            optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Recupera la inercia del optimizador
            epoca_inicial = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)


    # ==========================================
    # 5. ENTRENAMIENTO (El bucle principal de PyTorch)
    # ==========================================
    print(f"\n[3/6] Iniciando entrenamiento desde la época {epoca_inicial}...")

    if epoca_inicial < TOTAL_EPOCAS:
        if epoca_inicial == 0 and not os.path.exists(LOG_PATH):
            with open(LOG_PATH, mode='w', newline='') as f:
                csv.writer(f).writerow(['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss'])

        for epoch in range(epoca_inicial, TOTAL_EPOCAS):
            # --- FASE DE ENTRENAMIENTO ---
            model.train() # Activa comportamientos específicos para entrenamiento (como el Dropout)
            running_loss, correct, total = 0.0, 0, 0
            
            for imagenes, etiquetas in train_loader:
                imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
                
                # La "Trinidad" del entrenamiento en PyTorch:
                optimizer.zero_grad() # 1. Limpia los cálculos matemáticos residuales del paso anterior
                salidas = model(imagenes) # 2. Pasa las imágenes por la red (Forward Pass)
                loss = criterion(salidas, etiquetas) # Calcula qué tan equivocada estuvo la red
                loss.backward() # 3. Calcula en qué dirección deben cambiar los pesos (Backpropagation)
                
                optimizer.step() # Aplica los cambios matemáticos a las neuronas
                
                running_loss += loss.item() * imagenes.size(0)
                _, predicciones = torch.max(salidas, 1) # Obtiene la clase con la probabilidad más alta
                total += etiquetas.size(0)
                correct += (predicciones == etiquetas).sum().item()
                
            train_loss = running_loss / total
            train_acc = correct / total
            
            # --- FASE DE VALIDACIÓN ---
            model.eval() # Apaga el Dropout y congela estadísticas para que la evaluación sea justa
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad(): # Apaga el cálculo de gradientes para no gastar memoria innecesaria
                for imagenes, etiquetas in val_loader:
                    imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
                    salidas = model(imagenes)
                    loss = criterion(salidas, etiquetas)
                    
                    val_loss += loss.item() * imagenes.size(0)
                    _, predicciones = torch.max(salidas, 1)
                    val_total += etiquetas.size(0)
                    val_correct += (predicciones == etiquetas).sum().item()
                    
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            print(f"Época {epoch+1}/{TOTAL_EPOCAS} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
            
            # Registramos los resultados en el archivo CSV
            with open(LOG_PATH, mode='a', newline='') as f:
                csv.writer(f).writerow([epoch, train_acc, train_loss, val_acc, val_loss])
                
            # Evaluamos si este modelo es el mejor hasta ahora para guardar un Backup
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc
                }, BACKUP_MODEL_PATH)
                print(f"  -> val_accuracy mejoró. Modelo guardado en {BACKUP_MODEL_PATH}")
            else:
                patience_counter += 1 # Si no mejoró, aumentamos la impaciencia
                
            # Si pasaron N épocas sin mejorar, detenemos el bucle para evitar el sobreajuste (Overfitting)
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n Parada temprana (Early Stopping) en la época {epoch+1}.")
                # Restauramos los pesos de la mejor época antes de salir
                checkpoint = torch.load(BACKUP_MODEL_PATH, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                break

        print(f"Guardando el modelo maestro en '{FINAL_MODEL_PATH}'...")
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
    else:
        print("-> El modelo ya completó sus épocas previamente. Saltando entrenamiento.")


    # ==========================================
    # 6. EVALUACIÓN Y MATRIZ DE CONFUSIÓN
    # ==========================================
    print("\n[4/6] Evaluando en el set de pruebas (Test Dataset)...")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad(): # Nuevamente, sin gradientes para ahorrar memoria durante la inferencia
        for imagenes, etiquetas_reales in test_loader:
            imagenes = imagenes.to(device)
            predicciones_lote = model(imagenes)
            _, clases_predichas = torch.max(predicciones_lote, 1)
            
            # Guardamos las etiquetas reales y las predichas para armar la matriz comparativa
            y_true.extend(etiquetas_reales.numpy())
            y_pred.extend(clases_predichas.cpu().numpy()) # Mandamos a CPU antes de convertir a Numpy

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n[5/6] Generando Matriz de Confusión y Reporte...")
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=CM_FIGSIZE)
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.title(f'Matriz de Confusión - Pokédex ({MODEL_NAME} PyTorch/Timm)', fontsize=CM_TITLE_FONTSIZE)
    plt.ylabel('Clase Real', fontsize=CM_LABEL_FONTSIZE)
    plt.xlabel('Clase Predicha', fontsize=CM_LABEL_FONTSIZE)
    plt.savefig('matriz_confusion.png') 
    plt.close() # Libera la memoria gráfica
    print("-> Matriz de confusión guardada como 'matriz_confusion.png'.")


    # ==========================================
    # 7. PRUEBA DE LA POKÉDEX (PREDICCIÓN INDIVIDUAL)
    # ==========================================
    if os.path.exists(TEST_IMAGE_PATH):
        print(f"\n[6/6] Analizando la imagen '{TEST_IMAGE_PATH}'...")
        
        img = Image.open(TEST_IMAGE_PATH).convert('RGB')
        # unsqueeze(0) le agrega una dimensión vacía al inicio para engañar al modelo y que crea que es un "Lote" de 1 imagen
        img_tensor = val_test_transforms(img).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor) # La salida cruda (logits)
            probabilidades = torch.nn.functional.softmax(outputs, dim=1) # Convierte los logits en porcentajes reales (0 a 1)
            top_prob, predicted_idx = torch.max(probabilidades, 1) # Extrae el porcentaje más alto y su posición
            
        predicted_index = predicted_idx.item()
        top_probability = top_prob.item() * 100
        predicted_pokemon = class_names[predicted_index]

        print("\n===============================")
        print(f"POKÉDEX RESULTADO:")
        print(f"Pokémon detectado: {predicted_pokemon.upper()}")
        print(f"Nivel de confianza: {top_probability:.2f}%")
        print("===============================\n")

        # Configuración del gráfico visual de salida
        plt.figure(figsize=PRED_FIGSIZE)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Pokédex Data:\nEspecie: {predicted_pokemon.upper()}\nConfianza: {top_probability:.2f}%", 
                  fontsize=PRED_TITLE_FONTSIZE, fontweight='bold', color='darkred', loc='left')
        plt.tight_layout()
        
        plt.savefig(FINAL_VISUALIZATION_PATH) # Guardamos sin intentar mostrar en pantalla para evitar el error de entorno en Linux
        print(f"-> Imagen de visualización guardada como '{FINAL_VISUALIZATION_PATH}'.")
        plt.close()
    else:
        print(f"Aviso: No se encontró la imagen '{TEST_IMAGE_PATH}'. Pon una imagen en la carpeta para probar la predicción.")