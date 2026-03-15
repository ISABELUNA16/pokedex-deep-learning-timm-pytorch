import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# CONFIGURACIÓN DINÁMICA DE HARDWARE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

if torch.cuda.is_available():
    BATCH_SIZE = 16 
    print(f" GPU Detectada: {torch.cuda.get_device_name(0)}")
else:
    print(f" Entrenando en CPU.")

# ==========================================
# 1. PARÁMETROS GLOBALES
# ==========================================
IMG_SIZE = 224 
DATASET_PATH = Path("pokemon_datasets") / "pokemon150" 
BACKUP_MODEL_PATH = 'backup_pokedex_model.pth' 
FINAL_MODEL_PATH = 'pokedex150_timm_model.pth'
LOG_PATH = 'historial_entrenamiento.csv' 
DICCIONARIO_PATH = 'clases_pokedex.json'
TEST_IMAGE_PATH = 'pikachu_test.png' 
TOTAL_EPOCAS = 15

# ==========================================
# 2. CARGA Y TRANSFORMACIONES (DATA AUGMENTATION)
# ==========================================
print("[1/6] Preparando DataLoaders...")

# Transformaciones de entrenamiento (Data Augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformaciones de validación/test
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carga inicial para obtener clases
full_dataset = datasets.ImageFolder(DATASET_PATH)
class_names = full_dataset.classes
num_classes = len(class_names)

with open(DICCIONARIO_PATH, 'w') as f:
    json.dump({i: nombre for i, nombre in enumerate(class_names)}, f)

# División 80/10/10
indices = list(range(len(full_dataset)))
np.random.seed(123)
np.random.shuffle(indices)

train_idx, temp_idx = indices[:int(0.8*len(indices))], indices[int(0.8*len(indices)):]
val_idx, test_idx = temp_idx[:len(temp_idx)//2], temp_idx[len(temp_idx)//2:]

train_dataset = Subset(datasets.ImageFolder(DATASET_PATH, transform=train_transforms), train_idx)
val_dataset = Subset(datasets.ImageFolder(DATASET_PATH, transform=val_transforms), val_idx)
test_dataset = Subset(datasets.ImageFolder(DATASET_PATH, transform=val_transforms), test_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==========================================
# 4. LÓGICA DE MODELO CON TIMM
# ==========================================
print("\n[2/6] Configurando modelo con timm...")

# Aquí está la magia de timm: creamos la ResNet50V2 y cambiamos la salida en una línea
model = timm.create_model('resnetv2_50', pretrained=True, num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 5. ENTRENAMIENTO (FIT)
# ==========================================
print(f"\n[3/6] Iniciando entrenamiento...")

best_acc = 0.0
patience = 3
trigger_times = 0

if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'w') as f: f.write("epoch,accuracy,loss,val_accuracy,val_loss\n")

for epoch in range(TOTAL_EPOCAS):
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

    # Log y Checkpoint
    with open(LOG_PATH, 'a') as f: f.write(f"{epoch},{epoch_acc},{epoch_loss},{v_acc},{v_loss}\n")
    
    if v_acc > best_acc:
        best_acc = v_acc
        torch.save(model.state_dict(), BACKUP_MODEL_PATH)
        print(f"val_accuracy mejoró. Guardando en {BACKUP_MODEL_PATH}")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early Stopping!")
            break

torch.save(model.state_dict(), FINAL_MODEL_PATH)

# ==========================================
# 6. EVALUACIÓN Y MATRIZ
# ==========================================
print("\n[4/6] Evaluando Test Set...")
model.load_state_dict(torch.load(BACKUP_MODEL_PATH))
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
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.savefig('matriz_confusion_pytorch.png')

print("-> Matriz guardada.")

# ==========================================
# 7. PREDICCIÓN FINAL
# ==========================================
if os.path.exists(TEST_IMAGE_PATH):
    img = Image.open(TEST_IMAGE_PATH).convert('RGB')
    input_tensor = val_transforms(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        conf, idx = torch.max(prob, 1)
    
    print(f"\nPOKÉDEX RESULTADO: {class_names[idx.item()].upper()}")
    print(f"Confianza: {conf.item()*100:.2f}%")

    # Visualización de la interfaz Pokédex
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Pokédex Data:\nEspecie: {class_names[idx.item()].upper()}\nConfianza: {conf.item()*100:.2f}%", 
              fontsize=14, fontweight='bold', color='darkred', loc='left')
    plt.tight_layout()
    
    FINAL_VISUALIZATION_PATH = 'prediccion_final_pokedex.png'
    plt.savefig(FINAL_VISUALIZATION_PATH)
    print(f"-> Imagen de visualización guardada como '{FINAL_VISUALIZATION_PATH}'. Puedes abrirla para ver el resultado.")
    
    # Opcional: Cerrar la figura para liberar memoria
    plt.close()
    #plt.show()