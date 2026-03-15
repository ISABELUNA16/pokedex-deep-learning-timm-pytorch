import streamlit as st
import torch
import timm
import json
from PIL import Image
from torchvision import transforms

# Interfaz Web
PAGE_TITLE = "Pokédex AI"
PAGE_LAYOUT = "centered"
APP_TITLE = "Pokedex Lite"
APP_SUBTITLE = "Projecto: Pytorch Image Model (timm) \n\n\n Sube la imagen de un Pokemon y la red neuronal ResNet50V2 lo identificará."
ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg"]

# Archivos del Modelo
MODEL_PATH = 'pokedex150_timm_model.pth'
DICT_PATH = 'clases_pokedex.json'
MODEL_NAME = 'resnetv2_50'

# Preprocesamiento de Imágenes
IMG_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
TOP_K_PREDICTIONS = 3 # Cuántas predicciones mostrar en la interfaz

# Configuracion de la pagina
st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)
st.title(APP_TITLE)
st.write(APP_SUBTITLE)

# carga del modelo en cache
@st.cache_resource
def cargar_pokedex():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar el diccionario
    with open(DICT_PATH, 'r') as f:
        diccionario_str = json.load(f)
        class_names = {int(k): v for k, v in diccionario_str.items()}
    
    num_classes = len(class_names)
    
    # Reconstruir la arquitectura
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    
    # Cargar los pesos
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval() 
    
    return model, class_names, device

try:
    model, class_names, device = cargar_pokedex()
except Exception as e:
    st.error(f"Error cargando los archivos vitales. Asegúrate de que el .pth y el .json estén en esta carpeta. Detalle: {e}")
    st.stop()

# Pre procesamiento
transformacion = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])


# Interfaz y prediccion
st.markdown("---")
imagen_subida = st.file_uploader("Elige una foto de tu computadora", type=ALLOWED_IMAGE_TYPES)

if imagen_subida is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(imagen_subida).convert('RGB')
    with col1:
        st.image(image, caption='Sujeto a identificar', use_container_width=True)
    
    with col2:
        st.write("Analizando datos de la imagen... 🔍")
        
        input_tensor = transformacion(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilidades = torch.nn.functional.softmax(outputs, dim=1)
            
            # Extraemos el Top K de probabilidades e índices
            top_prob, top_indices = torch.topk(probabilidades, TOP_K_PREDICTIONS, dim=1)
            
        # Convertimos los tensores a listas normales de Python para iterar fácilmente
        top_prob = top_prob.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()
        
        # --- RESULTADO PRINCIPAL (Top 1) ---
        pokemon_principal = class_names[top_indices[0]].upper()
        confianza_principal = top_prob[0] * 100
        
        st.success(f"**Pokémon: {pokemon_principal}**")
        st.progress(int(confianza_principal)) 
        st.write(f"Nivel de confianza: **{confianza_principal:.2f}%**")
        
        # --- RESULTADOS SECUNDARIOS (Top 2 y Top 3) ---
        st.markdown("---")
        st.write("**Otras posibilidades:**")
        for i in range(1, TOP_K_PREDICTIONS):
            pokemon_secundario = class_names[top_indices[i]].capitalize()
            confianza_secundaria = top_prob[i] * 100
            st.write(f"- {pokemon_secundario}: *{confianza_secundaria:.2f}%*")