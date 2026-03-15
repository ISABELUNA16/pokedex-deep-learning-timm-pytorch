import os
import json
import csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm

warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")

# ==========================================
# CONFIGURACIÓN DE HARDWARE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 if torch.cuda.is_available() else 32

if torch.cuda.is_available():
    print(f"GPU Detectada: {torch.cuda.get_device_name(0)}")
else:
    print("Entrenando en CPU.")

# ==========================================
# 1. CONSTANTES GLOBALES
# ==========================================
# Rutas y Archivos
DATASET_PATH = Path("pokemon_datasets") / "pokemon150" 
BACKUP_MODEL_PATH = 'backup_pokedex_model.pth' 
FINAL_MODEL_PATH = 'pokedex150_timm_model.pth'
LOG_PATH = 'historial_entrenamiento.csv' 
DICCIONARIO_PATH = 'clases_pokedex.json'
TEST_IMAGE_PATH = 'pikachu_test.png' 
CM_SAVE_PATH = 'matriz_confusion_pytorch.png'
FINAL_VISUALIZATION_PATH = 'prediccion_final_pokedex.png'

# Hiperparámetros del Modelo y Entrenamiento
MODEL_NAME = 'resnetv2_50'
TOTAL_EPOCAS = 15
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 3

# Hiperparámetros del Scheduler (Reductor de Learning Rate)
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 1

# Procesamiento de Datos y Splits
IMG_SIZE = 224 
RANDOM_SEED = 123 #semilla
TRAIN_SPLIT_RATIO = 0.8 # 80% para Train (el 20% restante se divide en Val y Test)
NUM_WORKERS = 4

# Data Augmentation y Normalización
ROTATION_DEGREES = 20
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Configuración Visual (Matplotlib/Seaborn)
CM_FIGSIZE = (15, 12)
CM_CMAP = 'Blues'
PRED_FIGSIZE = (6, 6)
PRED_TITLE_FONTSIZE = 14
PRED_TITLE_FONTWEIGHT = 'bold'
PRED_TITLE_COLOR = 'darkred'
PRED_TITLE_LOC = 'left'

# 2. TRANSFORMACIONES
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(ROTATION_DEGREES),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# BLOQUE PRINCIPAL (Protección Multiprocessing)
if __name__ == '__main__':
    print("[1/6] Preparando DataLoaders...")

    full_dataset = datasets.ImageFolder(DATASET_PATH)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    with open(DICCIONARIO_PATH, 'w') as f:
        json.dump({i: nombre for i, nombre in enumerate(class_names)}, f)

    indices = list(range(len(full_dataset)))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)

    # División de datos usando la constante
    train_split_index = int(TRAIN_SPLIT_RATIO * len(indices))
    train_idx, temp_idx = indices[:train_split_index], indices[train_split_index:]
    val_idx, test_idx = temp_idx[:len(temp_idx)//2], temp_idx[len(temp_idx)//2:]

    train_dataset = Subset(datasets.ImageFolder(DATASET_PATH, transform=train_transforms), train_idx)
    val_dataset = Subset(datasets.ImageFolder(DATASET_PATH, transform=val_transforms), val_idx)
    test_dataset = Subset(datasets.ImageFolder(DATASET_PATH, transform=val_transforms), test_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    # 3. LÓGICA DE MODELO CON TIMM
    print("\n[2/6] Configurando modelo con timm...")

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=SCHEDULER_FACTOR, 
        patience=SCHEDULER_PATIENCE
    )

    # 4. LÓGICA DE SALTO O REANUDACIÓN
    print(f"\n[3/6] Iniciando fase de entrenamiento...")
    
    epoca_inicial = 0
    best_acc = 0.0
    patience_counter = 0

    if os.path.exists(FINAL_MODEL_PATH):
        print("¡Modelo final detectado! Saltando el entrenamiento...")
        model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device, weights_only=True))
        epoca_inicial = TOTAL_EPOCAS 
    elif os.path.exists(BACKUP_MODEL_PATH):
        print("Retomando desde el backup...")
        checkpoint = torch.load(BACKUP_MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoca_inicial = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)

    # 5. ENTRENAMIENTO (FIT)
    if epoca_inicial < TOTAL_EPOCAS:
        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, 'w') as f: f.write("epoch,accuracy,loss,val_accuracy,val_loss\n")

        for epoch in range(epoca_inicial, TOTAL_EPOCAS):
            # FASE TRAIN
            model.train()
            train_loss, train_correct = 0.0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            # FASE VAL
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
            
            epoch_loss = train_loss / len(train_idx)
            epoch_acc = train_correct / len(train_idx)
            v_loss = val_loss / len(val_idx)
            v_acc = val_correct / len(val_idx)

            print(f"Época {epoch+1}/{TOTAL_EPOCAS} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - val_loss: {v_loss:.4f} - val_acc: {v_acc:.4f}")
            
            scheduler.step(v_loss)

            with open(LOG_PATH, 'a') as f: f.write(f"{epoch},{epoch_acc},{epoch_loss},{v_acc},{v_loss}\n")
            
            if v_acc > best_acc:
                best_acc = v_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc
                }, BACKUP_MODEL_PATH)
                print(f"val_accuracy mejoró. Guardando en {BACKUP_MODEL_PATH}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print("Early Stopping!")
                    checkpoint = torch.load(BACKUP_MODEL_PATH, map_location=device, weights_only=True)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    break

        torch.save(model.state_dict(), FINAL_MODEL_PATH)

    # 6. EVALUACIÓN Y MATRIZ
    print("\n[4/6] Evaluando Test Set...")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_true.extend(labels.numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())

    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=CM_FIGSIZE)
    sns.heatmap(cm, annot=False, cmap=CM_CMAP)
    plt.savefig(CM_SAVE_PATH)
    plt.close()
    print("-> Matriz guardada.")

    # 7. PREDICCIÓN FINAL
    if os.path.exists(TEST_IMAGE_PATH):
        img = Image.open(TEST_IMAGE_PATH).convert('RGB')
        input_tensor = val_transforms(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, idx = torch.max(prob, 1)
        
        print(f"\nPOKÉDEX RESULTADO: {class_names[idx.item()].upper()}")
        print(f"Confianza: {conf.item()*100:.2f}%")

        plt.figure(figsize=PRED_FIGSIZE)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Pokédex Data:\nEspecie: {class_names[idx.item()].upper()}\nConfianza: {conf.item()*100:.2f}%", 
                  fontsize=PRED_TITLE_FONTSIZE, fontweight=PRED_TITLE_FONTWEIGHT, color=PRED_TITLE_COLOR, loc=PRED_TITLE_LOC)
        plt.tight_layout()
        
        plt.savefig(FINAL_VISUALIZATION_PATH)
        print(f"-> Imagen guardada como '{FINAL_VISUALIZATION_PATH}'.")
        plt.close()