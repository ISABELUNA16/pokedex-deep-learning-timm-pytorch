import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import kagglehub
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image


# DESCARGA Y PREPARACIÓN DEL DATASET

print("Descargando el dataset de Kaggle...")
path = kagglehub.dataset_download("noodulz/pokemon-dataset-1000")

# Buscador de la carpeta correcta con las imágenes
data_dir = path
for raiz, directorios, archivos in os.walk(path):
    if len(directorios) > 100: # Buscamos la carpeta con las subcarpetas de los Pokémon
        data_dir = raiz
        break

print(f"Carpeta principal encontrada en: {data_dir}")

# ResNetV2 espera imágenes de 224x224 para funcionar al máximo de su capacidad
IMG_SIZE = 224 
BATCH_SIZE = 32

print("Cargando imágenes de entrenamiento y validación...")

# 1. Primero cargamos los datasets

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)


# 2. ¡CORRECCIÓN AQUÍ! Extraemos los nombres ANTES de optimizar
nombres_clases = train_dataset.class_names
num_classes = len(nombres_clases)

# 3. AHORA aplicamos la optimización de carga en memoria
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

ruta_modelo = 'pokedex_resnet152v2_noodulz.keras'


# ---------------------------------------------------------
# 2. ARQUITECTURA DEL MODELO Y DATA AUGMENTATION
# ---------------------------------------------------------
if os.path.exists(ruta_modelo):
    print("¡Modelo encontrado! Cargando la Pokédex entrenada...")
    model = tf.keras.models.load_model(ruta_modelo)
else:
    print("Construyendo la arquitectura de la Pokédex (ResNet152V2)...")
    
    # Bloque de Data Augmentation: Vital para datasets pequeños
    # data_augmentation = tf.keras.Sequential([
    #    layers.RandomFlip("horizontal"),
    #    layers.RandomRotation(0.2),
    #    layers.RandomZoom(0.2),
    #    layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
    # ], name="data_augmentation")

    # Modelo base de Keras (congelado por ahora)
    # ResNet152V2 = version mas robusta
    # ResNet50V2 = light version 
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False 

    # Ensamblaje del modelo
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # 1. Aumentamos los datos (solo ocurre durante el entrenamiento)
        # data_augmentation,
        
        # 2. ResNetV2 exige que los píxeles estén escalados entre -1 y 1
        layers.Rescaling(1./127.5, offset=-1),
        
        # 3. Extractor de características
        base_model,
        
        # 4. Clasificador final
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5), # Dropout muy alto (50%) para combatir el sobreajuste
        layers.Dense(num_classes, activation='softmax')
    ])

    # ---------------------------------------------------------
    # 3. COMPILACIÓN Y ENTRENAMIENTO
    # ---------------------------------------------------------
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Parada temprana para evitar entrenar de más si ya aprendió todo lo posible
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    print("Iniciando entrenamiento...")
    history = model.fit(
        train_dataset, 
        epochs=3, 
        validation_data=validation_dataset,
        callbacks=[early_stopping]
    )
    
    print(f"Guardando el modelo maestro en '{ruta_modelo}'...")
    model.save(ruta_modelo)

# ---------------------------------------------------------
# 4. PRUEBA DE LA POKÉDEX (PREDICCIÓN)
# ---------------------------------------------------------
# Reemplaza esto con el nombre de cualquier imagen que tengas en tu computadora
ruta_nueva_imagen = 'pikachu.png' 

if os.path.exists(ruta_nueva_imagen):
    print(f"\nAnalizando la imagen '{ruta_nueva_imagen}'...")
    
    # Preparar la imagen
    img = image.load_img(ruta_nueva_imagen, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    predicciones = model.predict(img_array)
    indice_ganador = np.argmax(predicciones[0])
    confianza = predicciones[0][indice_ganador] * 100

    nombre_pokemon = nombres_clases[indice_ganador]

    # Visualización de la interfaz Pokédex
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Pokédex Data:\nEspecie: {nombre_pokemon.upper()}\nConfianza: {confianza:.2f}%", 
              fontsize=14, fontweight='bold', color='darkred', loc='left')
    plt.tight_layout()
    plt.show()
else:
    print(f"Aviso: No se encontró la imagen '{ruta_nueva_imagen}'. Pon una imagen en la carpeta para probar la predicción.")