import os
# El nivel '3' bloquea mensajes de INFO, WARNINGS y ERRORS del backend de C++
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path


# CONFIGURACIÓN DINÁMICA DE HARDWARE (CPU vs GPU)
BATCH_SIZE = 32 # Lote por defecto si no hay GPU

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Reducimos el lote a 16 para los 4GB de VRAM de la Quadro T2000
        BATCH_SIZE = 16 
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detectada: {gpus[0].name}")
        print(f"✅ Memoria dinámica activada. BATCH_SIZE ajustado a {BATCH_SIZE}.")
    except RuntimeError as e:
        print(f"⚠ Error configurando la GPU: {e}")
else:
    print(f"💻 Entrenando en CPU. BATCH_SIZE ajustado a {BATCH_SIZE}.")

# ==========================================
# 1. PARÁMETROS GLOBALES Y DESCARGA
# ==========================================

IMG_SIZE = 224 
NUM_CLASSES = 151 # No se usa
DATASET_PATH = Path("pokemon_datasets") / "pokemon150" # Editar cuando se cambia de dataset, apuntar a la carpeta que tiene a los pokemon por carpeta
BACKUP_MODEL_PATH = 'backup_pokedex_model.h5' # Archivo del modelo auxiliar en caso de interruciones
FINAL_MODEL_PATH = 'pokedex150_resnet50v2_model.keras'
LOG_PATH = 'historial_entrenamiento.csv' # Log del entrenamiento, usado para recuperaciones
DICCIONARIO_PATH = 'clases_pokedex.json'
TEST_IMAGE_PATH = 'pikachu_test.png' 
TOTAL_EPOCAS = 20

data_dir = DATASET_PATH
print(f"Carpeta principal encontrada en: {data_dir}")


# ==========================================
# 2. CARGA Y DIVISIÓN DE DATOS (TRAIN, VAL, TEST)
# ==========================================

print("[1/6] Cargando imágenes para entrenamiento, validación y test...")

val_dataset_inicial = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int' # Formato óptimo (Sparse)
)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int' # Formato óptimo (Sparse)
)


# Guardamos los nombres de las clases en un JSON
class_names = train_dataset.class_names
with open(DICCIONARIO_PATH, 'w') as f:
    json.dump({i: nombre for i, nombre in enumerate(class_names)}, f)
    
num_classes = len(class_names)


# Dividimos el val_dataset_inicial a la mitad (10% Validación, 10% Test)
batches_totales = tf.data.experimental.cardinality(val_dataset_inicial)
val_batches = batches_totales // 2

val_dataset = val_dataset_inicial.take(val_batches)
test_dataset = val_dataset_inicial.skip(val_batches)


# Ahora aplicamos la optimización de carga en memoria
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# ==========================================
# 3. CONFIGURACIÓN DE CALLBACKS
# ==========================================

# Definimos el primer callback (Guardar el mejor modelo)
checkpoint = tf.keras.callbacks.ModelCheckpoint(BACKUP_MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
# El segundo callback hace una parada temprana para evitar entrenar de más si ya aprendió todo lo posible
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
# Este callback anota cada época terminada en un archivo de texto. 
# append=True asegura que si retomamos, siga escribiendo debajo.
csv_logger = tf.keras.callbacks.CSVLogger(LOG_PATH, append=True)



# ==========================================
# 4. LÓGICA DE MODELO (CREAR O RETOMAR)
# ==========================================
# Modelo base de Keras (congelado por ahora)
    # ResNet152V2 = version mas robusta
    # ResNet50V2 = light version - adecuada para este caso
print("\n[2/6] Verificando estado del modelo...")

if os.path.exists(FINAL_MODEL_PATH):
    print("¡Modelo encontrado! Cargando la Pokédex entrenada...")
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
else:
    print("Construyendo la arquitectura de la Pokédex (ResNet152V2)...")

    epoca_inicial = 0

    if os.path.exists(BACKUP_MODEL_PATH):
        print(f"-> Retomando modelo existente: {BACKUP_MODEL_PATH}")
        model = tf.keras.models.load_model(BACKUP_MODEL_PATH)
        if os.path.exists(LOG_PATH):
            try:
                df_log = pd.read_csv(LOG_PATH)
                epoca_inicial = df_log['epoch'].max() + 1
            except:
                pass
    else:
        print("-> Creando arquitectura ResNet50V2 desde cero...")
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ])

        base_model = tf.keras.applications.ResNet50V2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3), 
            include_top=False, # El "top" es la última capa de la red ("la cabeza"). Con False, la cortamos y nos quedamos solo con el "cuerpo", que es un experto extractor de texturas, formas y colores.
            weights='imagenet' # Descarga los pesos matemáticos reales preentrenados por Google
        )
        base_model.trainable = False # Congelamos esta red ("el cerebro"). 
        # Le decimos: "Tus conocimientos ya son perfectos, no quiero que el entrenamiento modifique tus pesos y arruine lo que ya aprendiste".

        inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = data_augmentation(inputs)
        x = tf.keras.applications.resnet_v2.preprocess_input(x) # Hace el reescalado -> layers.Rescaling(1./127.5, offset=-1),  es menos manual y mas sostenible
        x = base_model(x, training=False) # Pasaremos las imagenes por el cerebro de ResNet50V2 para que extraiga las características (orejas puntiagudas, color amarillo, cola en zigzag).
        x = tf.keras.layers.GlobalAveragePooling2D()(x) # Aplasta ese bloque 3D y lo convierte en un vector 1D (una lista simple de 2048 números)
        x = tf.keras.layers.Dropout(0.5)(x) # Durante el entrenamiento, apaga (hace cero) al 50% de las neuronas aleatoriamente en cada paso. Obliga a la red a no depender de una sola característica
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x) # La capa final de decisión. Conecta esa información resumida a num_classes neuronas, donde num_classes es la cantidad de pokemones

        model = tf.keras.models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # ==========================================
    # 5. ENTRENAMIENTO
    # ==========================================
    print(f"\n[3/6] Iniciando entrenamiento desde la época {epoca_inicial}...")
    if epoca_inicial < TOTAL_EPOCAS:
        model.fit(
            train_dataset,
            epochs=TOTAL_EPOCAS,
            initial_epoch=epoca_inicial,
            validation_data=val_dataset,
            callbacks=[checkpoint, early_stopping, csv_logger]
        )

        print(f"Guardando el modelo maestro en '{FINAL_MODEL_PATH}'...")
        model.save(FINAL_MODEL_PATH)
    else:
        print("-> El modelo ya completó sus épocas previamente.")
    

    # ==========================================
    # 6. EVALUACIÓN Y MATRIZ DE CONFUSIÓN
    # ==========================================
    print("\n[4/6] Evaluando en el set de pruebas (Test Dataset)...")
    y_true, y_pred = [], []

    # Iteramos sobre los lotes (batches) del test_dataset
    for imagenes, etiquetas_reales in test_dataset:
        # El modelo hace sus predicciones para este lote de 32 imágenes
        predicciones_lote = model.predict(imagenes, verbose=0)
        # np.argmax nos da el índice con la probabilidad más alta
        clases_predichas = np.argmax(predicciones_lote, axis=1)
        # Guardamos los resultados para armar la matriz después
        y_true.extend(etiquetas_reales.numpy())
        y_pred.extend(clases_predichas)

    # Convertimos las listas a arreglos de Numpy para que scikit-learn las entienda
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n[5/6] Generando Matriz de Confusión y Reporte...")
    print("\n--- Reporte de Clasificación ---")
    # Calcular metricas
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Calcular la Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.title('Matriz de Confusión - Pokédex (ResNet50V2)', fontsize=16)
    plt.ylabel('Clase Real', fontsize=12)
    plt.xlabel('Clase Predicha', fontsize=12)
    plt.savefig('matriz_confusion.png') # Guardamos el gráfico en disco
    plt.close()
    print("-> Matriz de confusión guardada como 'matriz_confusion.png'.")

    

# ---------------------------------------------------------
# 4. PRUEBA DE LA POKÉDEX (PREDICCIÓN)
# ---------------------------------------------------------
# Reemplaza esto con el nombre de cualquier imagen que tengas en tu computadora

if os.path.exists(TEST_IMAGE_PATH):
    print(f"\nAnalizando la imagen '{TEST_IMAGE_PATH}'...")
    
    # Cargamos la imagen y forzamos el tamaño a 224x224 (el input de ResNet50V2)
    img = tf.keras.utils.load_img(TEST_IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))

    # Convertimos la imagen a un arreglo matemático de Numpy (forma: 224, 224, 3)
    img_array = tf.keras.utils.img_to_array(img)

    # La red neuronal no espera una sola imagen, espera un "lote" (batch).
    # Usamos expand_dims para agregar una dimensión falsa al inicio. Forma final: (1, 224, 224, 3)
    img_array = tf.expand_dims(img_array, 0) 


    # 5. Hacer la predicción
    print("Analizando la nueva imagen...")
    predictions = model.predict(img_array)

    # Predicciones es un arreglo con num_classes probabilidades (cantidad de pokemons). 
    # np.argmax busca el índice del número más alto (la probabilidad mayor)
    predicted_index = np.argmax(predictions[0])
    top_probability = predictions[0][predicted_index] * 100

    # Usamos el índice para buscar el nombre del Pokémon en el diccionario
    # json.load lee las llaves como strings, por eso convertimos el índice a string
    predicted_pokemon = class_names[predicted_index]

    print("\n===============================")
    print(f"POKÉDEX RESULTADO:")
    print(f"Pokémon detectado: {predicted_pokemon.upper()}")
    print(f"Nivel de confianza: {top_probability:.2f}%")
    print("===============================\n")

    # Visualización de la interfaz Pokédex
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Pokédex Data:\nEspecie: {predicted_pokemon.upper()}\nConfianza: {top_probability:.2f}%", 
              fontsize=14, fontweight='bold', color='darkred', loc='left')
    plt.tight_layout()
    #plt.show()
    plt.savefig("prediction_image_result.png", bbox_inches='tight')
    print("✅ Imagen del resultado guardada como 'prediction_image_result.png'")
else:
    print(f"Aviso: No se encontró la imagen '{TEST_IMAGE_PATH}'. Pon una imagen en la carpeta para probar la predicción.")