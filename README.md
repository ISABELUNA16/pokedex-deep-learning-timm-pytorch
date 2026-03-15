# 📱 PokedexLite: Clasificador de Pokemon con Deep Learning - Timm (Pythorch Image Models)

Este proyecto es una Pokedex inteligente desarrollada en **Ubuntu** que utiliza modelos de visión por computadora de última generación (librería `timm`). Permite entrenar y ejecutar inferencia para identificar especies de Pokemon a partir de imágenes.

## 📁 Estructura del Repositorio

- `app.py`: Aplicación web con interfaz de usuario (Streamlit)
- `pokedex150_timm_fullfinetuning.py`: Script principal (entrenamiento e inferencia).
- `pokemon_datasets/`: Carpeta para las imágenes (mantenida mediante `.gitkeep`).
- `requirements_gpu.txt`: Dependencias para entornos con aceleración CUDA (NVIDIA).
- `requirements_cpu.txt`: Dependencias ligeras para ejecución en procesador.
- `pokedex_model.pth`: Pesos del modelo entrenado listos para usar.
- `clases_pokedex.json`: Mapeo de índices a nombres reales de Pokémon (Esencial para la App).

---

## 🛠️ Configuración e Instalación

### 1. Clonar y Preparar Estructura
```bash
git clone [https://github.com/ISABELUNA16/pokedex-deep-learning-timm-pytorch.git](https://github.com/ISABELUNA16/pokedex-deep-learning-timm-pytorch.git)
cd pokedex-deep-learning
```

## 2. Gestión de Entornos Virtuales (Ubuntu)

Dependiendo del hardware, configura tu entorno adecuado siguiendo estas instrucciones exactas de terminal:

### 🚀 Opción A: Con GPU (Recomendado para entrenamiento)
Ideal si usas **Ubuntu** o **WSL2** con una tarjeta NVIDIA y drivers CUDA instalados.

```bash
# Crear el entorno virtual
python3 -m venv venv_wsl

# Activar el entorno
source venv_wsl/bin/activate

# Instalar dependencias para GPU
pip install -r requirements_gpu.txt
```
### 🚀 Opción B: Con CPU (Ideal para inferencia o laptops básicas)
Ideal si usas **Ubuntu** o **WSL2** con una tarjeta NVIDIA y drivers CUDA instalados.

```bash
# Crear el entorno virtual
python3 -m venv venv

# Activar el entorno
source venv/bin/activate

# Instalar dependencias para CPU
pip install -r requirements_cpu.txt
```
## 3. Motor de Deep Learning

Este script es el corazón del proyecto. Utiliza **Transfer Learning** con una arquitectura `resnetv2_50` preentrenada, optimizada para reconocer Pokemon con alta precisión.

### 1. Preparación previa
Asegúrate de tener tus imágenes organizadas por carpetas dentro de `pokemon_datasets/pokemon150/`. Por ejemplo:
- `pokemon_datasets/pokemon150/Pikachu/img1.jpg`
- `pokemon_datasets/pokemon150/Bulbasaur/img2.jpg`

### 2. Ejecución del Entrenamiento
Para iniciar el proceso de aprendizaje, simplemente activa tu entorno y ejecuta:

```bash
python3 pokedex150_timm_fullfinetuning.py
```

## 4. Interfaz Web (`app.py`)

Una vez finalizado el entrenamiento, puedes levantar la aplicación visual para probar tu Pokedex con cualquier imagen o mediante la cámara.

### Ejecución del Servidor
```bash
streamlit run app.py
```