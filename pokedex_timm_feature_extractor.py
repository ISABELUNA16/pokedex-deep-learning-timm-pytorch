import os
import json
import csv
import warnings
import numpy as np
import pandas as pd
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

# CONFIGURACIÓN DE HARDWARE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 if torch.cuda.is_available() else 32

if torch.cuda.is_available():
    print(f"GPU Detectada: {torch.cuda.get_device_name(0)}")
else:
    print("Entrenando en CPU.")

# 1. CONSTANTES Y PARÁMETROS GLOBALES
DATASET_PATH = Path("pokemon_datasets") / "pokemon150" 
BACKUP_MODEL_PATH = 'backup_pokedex_model.pth' 
FINAL_MODEL_PATH = 'pokedex150_resnet50v2_model.pth' 
LOG_PATH = 'historial_entrenamiento.csv' 
DICCIONARIO_PATH = 'clases_pokedex.json' 
TEST_IMAGE_PATH = 'pikachu_test.png' 
FINAL_VISUALIZATION_PATH = 'prediccion_final_pokedex.png' 

MODEL_NAME = 'resnetv2_50' 
TOTAL_EPOCAS = 20 
LEARNING_RATE = 0.001 
EARLY_STOPPING_PATIENCE = 3 

IMG_SIZE = 224 
VAL_TEST_SPLIT_RATIO = 0.2  
RANDOM_SEED = 123 
NUM_WORKERS = 4 

NORMALIZE_MEAN = [0.485, 0.456, 0.406] 
NORMALIZE_STD = [0.229, 0.224, 0.225]
FLIP_PROBABILITY = 0.5 
ROTATION_DEGREES = 36 
AFFINE_DEGREES = 0 
AFFINE_SCALE = (0.8, 1.2) 

CM_FIGSIZE = (20, 18) 
CM_TITLE_FONTSIZE = 16 
CM_LABEL_FONTSIZE = 12 
PRED_FIGSIZE = (6, 6) 
PRED_TITLE_FONTSIZE = 14 

# PIPELINES DE TRANSFORMACIÓN
val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
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

# BLOQUE PRINCIPAL
if __name__ == '__main__':
    # 2. CARGA Y DIVISIÓN DE DATOS
    full_dataset_train = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
    full_dataset_val = datasets.ImageFolder(DATASET_PATH, transform=val_test_transforms)

    class_names = full_dataset_train.classes
    with open(DICCIONARIO_PATH, 'w') as f:
        json.dump({i: nombre for i, nombre in enumerate(class_names)}, f)
    num_classes = len(class_names)

    total_size = len(full_dataset_train)
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(RANDOM_SEED)).tolist()

    val_split = int(VAL_TEST_SPLIT_RATIO * total_size)
    train_indices = indices[val_split:] 
    val_test_indices = indices[:val_split] 

    val_size = len(val_test_indices) // 2
    val_indices = val_test_indices[:val_size] 
    test_indices = val_test_indices[val_size:] 

    train_dataset = Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_val, val_indices)
    test_dataset = Subset(full_dataset_val, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 3 Y 4. LÓGICA DE MODELO
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    epoca_inicial = 0
    best_val_acc = 0.0
    patience_counter = 0 

    if os.path.exists(FINAL_MODEL_PATH):
        print("Cargando modelo final existente...")
        model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device, weights_only=True))
        epoca_inicial = TOTAL_EPOCAS 
    elif os.path.exists(BACKUP_MODEL_PATH):
        print("Retomando desde backup...")
        checkpoint = torch.load(BACKUP_MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict']) 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        epoca_inicial = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)

    # 5. ENTRENAMIENTO
    if epoca_inicial < TOTAL_EPOCAS:
        if epoca_inicial == 0 and not os.path.exists(LOG_PATH):
            with open(LOG_PATH, mode='w', newline='') as f:
                csv.writer(f).writerow(['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss'])

        for epoch in range(epoca_inicial, TOTAL_EPOCAS):
            # --- TRAIN ---
            model.train() 
            running_loss, correct, total = 0.0, 0, 0
            
            for imagenes, etiquetas in train_loader:
                imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
                
                optimizer.zero_grad() 
                salidas = model(imagenes) 
                loss = criterion(salidas, etiquetas) 
                loss.backward() 
                optimizer.step() 
                
                running_loss += loss.item() * imagenes.size(0)
                _, predicciones = torch.max(salidas, 1) 
                total += etiquetas.size(0)
                correct += (predicciones == etiquetas).sum().item()
                
            train_loss = running_loss / total
            train_acc = correct / total
            
            # --- VALIDACIÓN ---
            model.eval() 
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad(): 
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
            
            with open(LOG_PATH, mode='a', newline='') as f:
                csv.writer(f).writerow([epoch, train_acc, train_loss, val_acc, val_loss])
                
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc
                }, BACKUP_MODEL_PATH)
            else:
                patience_counter += 1 
                
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early Stopping en época {epoch+1}.")
                checkpoint = torch.load(BACKUP_MODEL_PATH, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                break

        torch.save(model.state_dict(), FINAL_MODEL_PATH)

    # 6. EVALUACIÓN Y MATRIZ DE CONFUSIÓN
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad(): 
        for imagenes, etiquetas_reales in test_loader:
            imagenes = imagenes.to(device)
            predicciones_lote = model(imagenes)
            _, clases_predichas = torch.max(predicciones_lote, 1)
            
            y_true.extend(etiquetas_reales.numpy())
            y_pred.extend(clases_predichas.cpu().numpy()) 

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=CM_FIGSIZE)
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.title(f'Matriz de Confusión - Pokédex ({MODEL_NAME})', fontsize=CM_TITLE_FONTSIZE)
    plt.ylabel('Clase Real', fontsize=CM_LABEL_FONTSIZE)
    plt.xlabel('Clase Predicha', fontsize=CM_LABEL_FONTSIZE)
    plt.savefig('matriz_confusion.png') 
    plt.close() 

    # 7. PRUEBA DE LA POKÉDEX (PREDICCIÓN)
    if os.path.exists(TEST_IMAGE_PATH):
        img = Image.open(TEST_IMAGE_PATH).convert('RGB')
        img_tensor = val_test_transforms(img).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor) 
            probabilidades = torch.nn.functional.softmax(outputs, dim=1) 
            top_prob, predicted_idx = torch.max(probabilidades, 1) 
            
        predicted_index = predicted_idx.item()
        top_probability = top_prob.item() * 100
        predicted_pokemon = class_names[predicted_index]

        print(f"\nPOKÉDEX RESULTADO: {predicted_pokemon.upper()} ({top_probability:.2f}%)")

        plt.figure(figsize=PRED_FIGSIZE)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Pokédex Data:\nEspecie: {predicted_pokemon.upper()}\nConfianza: {top_probability:.2f}%", 
                  fontsize=PRED_TITLE_FONTSIZE, fontweight='bold', color='darkred', loc='left')
        plt.tight_layout()
        
        plt.savefig(FINAL_VISUALIZATION_PATH) 
        plt.close()