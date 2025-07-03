
from PIL import Image, ImageDraw, ImageFont
import os
import shutil
import streamlit as st
import json
import subprocess
import numpy as np
from bert_score import BERTScorer
import random
import time
from deep_translator import GoogleTranslator
import requests
from fonction import get_best_reformulations,est_en_francais,traduire_en_anglais,segment_objects_with_prompting,compare_keywords_with_class,highlight_objects_in_image
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from PIL import Image
import io
import zipfile

st.set_page_config(page_title="SegmaVision Pro Light", layout="wide")

# --- Initialisation de l'état de la session ---
# Ce bloc garantit que toutes les variables existent dès le premier chargement de l'application.

# 1. Gestion de la navigation et des vues
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'main'  # Peut être 'main', 'about', 'prompt', 'enhancer'

# 2. Flux de la saisie de prompt
if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = ""
if 'user_prompt_english' not in st.session_state:
    st.session_state.user_prompt_english = ""
if 'user_prompt_original' not in st.session_state:
    st.session_state.user_prompt_original = ""
if 'prompt_submitted' not in st.session_state:
    st.session_state.prompt_submitted = False
if 'prompt_confirmed' not in st.session_state:
    st.session_state.prompt_confirmed = False
if 'prompt_error' not in st.session_state:
    st.session_state.prompt_error = False

# 3. Flux de l'amélioration de la prompt (Enhancer)
if 'enhance_mode' not in st.session_state:
    st.session_state.enhance_mode = False
if 'enhanced_prompts' not in st.session_state:
    st.session_state.enhanced_prompts = []
if 'selected_prompt_index' not in st.session_state:
    st.session_state.selected_prompt_index = -1  # -1 signifie aucune sélection
if 'enhance_time' not in st.session_state:
    st.session_state.enhance_time = ""

# 4. Flux de l'extraction et classification des mots-clés
if 'keywords_extracted' not in st.session_state:
    st.session_state.keywords_extracted = False
if 'extracted_keywords' not in st.session_state:
    st.session_state.extracted_keywords = []
if 'keywords_classified' not in st.session_state:
    st.session_state.keywords_classified = False
if 'objets_existants' not in st.session_state:
    st.session_state.objets_existants = []
if 'objets_inexistants' not in st.session_state:
    st.session_state.objets_inexistants = []
if 'sous_classes' not in st.session_state:
    st.session_state.sous_classes = []
if 'extraction_time' not in st.session_state:
    st.session_state.extraction_time = ""

# 5. Sélection de l'utilisateur pour le traitement
if 'selected_for_segmentation' not in st.session_state:
    st.session_state.selected_for_segmentation = []
if 'selected_for_finetuning' not in st.session_state:
    st.session_state.selected_for_finetuning = []
if 'selection_confirmed' not in st.session_state:
    st.session_state.selection_confirmed = False

# 6. Gestion des images et des résultats de segmentation
if 'show_upload' not in st.session_state:
    st.session_state.show_upload = False
if 'saved_images_paths' not in st.session_state:
    st.session_state.saved_images_paths = []  # Liste des chemins des fichiers sauvegardés
if 'active_image_index' not in st.session_state:
    st.session_state.active_image_index = 0  # Index de l'image sélectionnée par l'utilisateur
if 'segmented_results' not in st.session_state:
    st.session_state.segmented_results = {} # L'image résultat après segmentation
if 'segmentation_completed' not in st.session_state:
    st.session_state.segmentation_completed = False

if 'finetuning_class_mapping' not in st.session_state:
    st.session_state.finetuning_class_mapping = {}
if 'finetuning_all_labels' not in st.session_state:
    # Dictionnaire pour stocker les labels par image: {'path/to/img1': [labels], ...}
    st.session_state.finetuning_all_labels = {}
if 'finetuning_active_image_path' not in st.session_state:
    st.session_state.finetuning_active_image_path = None
if 'dataset_generated' not in st.session_state:
    st.session_state.dataset_generated = False


# Fonctions pour mettre à jour l'état

def show_about():
    st.session_state.current_view = 'about'

def show_prompt():
    st.session_state.current_view = 'prompt'
    st.session_state.prompt_submitted = False
    st.session_state.prompt_confirmed = False
    st.session_state.enhance_mode = False

def show_enhancer():
    if not st.session_state.user_prompt:
        st.session_state.current_view = 'prompt'
        st.session_state.prompt_error = True
    else:
        st.session_state.current_view = 'enhancer'
        enhance_prompt()

def submit_prompt():
    if st.session_state.user_prompt:
        st.session_state.prompt_submitted = True
        st.session_state.prompt_error = False
    else:
        st.session_state.prompt_error = True


def confirm_prompt():
    # Vérifier si la prompt est en français et la traduire si nécessaire
    if est_en_francais(st.session_state.user_prompt):
        with st.spinner("Traduction de la prompt en anglais..."):
            st.session_state.user_prompt_english = traduire_en_anglais(st.session_state.user_prompt)
            # Stocker la prompt originale pour l'affichage
            st.session_state.user_prompt_original = st.session_state.user_prompt
    else:
        # Si déjà en anglais, utiliser telle quelle
        st.session_state.user_prompt_english = st.session_state.user_prompt
        st.session_state.user_prompt_original = st.session_state.user_prompt

    st.session_state.prompt_confirmed = True
    st.session_state.enhance_mode = False
    # Réinitialiser les mots-clés extraits
    st.session_state.keywords_extracted = False
    st.session_state.extracted_keywords = []
    # Réinitialiser les classifications
    st.session_state.keywords_classified = False
    st.session_state.objets_existants = []
    st.session_state.objets_inexistants = []
    st.session_state.sous_classes = []

def start_enhance_mode():
    st.session_state.enhance_mode = True

def enhance_prompt():
    with st.spinner("💬 Amélioration de la prompt en cours..."):
        start_time = time.time()
        results = get_best_reformulations(st.session_state.user_prompt)
        end_time = time.time()

        if results:
            st.session_state.enhanced_prompts = results
            st.session_state.enhance_time = f"{end_time - start_time:.2f}"
            st.session_state.selected_prompt_index = -1  # Réinitialiser la sélection
            # Réinitialiser les mots-clés extraits
            st.session_state.keywords_extracted = False
            st.session_state.extracted_keywords = []
            # Réinitialiser les classifications
            st.session_state.keywords_classified = False
            st.session_state.objets_existants = []
            st.session_state.objets_inexistants = []
            st.session_state.sous_classes = []
        else:
            st.session_state.enhanced_prompts = []

def select_enhanced_prompt(index):
    st.session_state.selected_prompt_index = index
    # Réinitialiser les mots-clés extraits quand on change de prompt
    st.session_state.keywords_extracted = False
    st.session_state.extracted_keywords = []
    # Réinitialiser les classifications
    st.session_state.keywords_classified = False
    st.session_state.objets_existants = []
    st.session_state.objets_inexistants = []
    st.session_state.sous_classes = []

def confirm_selection():
    if st.session_state.selected_prompt_index >= 0:
        # Utiliser la version anglaise pour le traitement
        st.session_state.user_prompt_english = st.session_state.enhanced_prompts[st.session_state.selected_prompt_index][0]
        # Stocker la prompt originale en français pour l'affichage
        st.session_state.user_prompt_original = st.session_state.user_prompt
        st.session_state.current_view = 'prompt'
        st.session_state.prompt_submitted = True
        st.session_state.prompt_confirmed = True  # Auto-confirmer la prompt sélectionnée
        # Réinitialiser les mots-clés extraits
        st.session_state.keywords_extracted = False
        st.session_state.extracted_keywords = []
        # Réinitialiser les classifications
        st.session_state.keywords_classified = False
        st.session_state.objets_existants = []
        st.session_state.objets_inexistants = []
        st.session_state.sous_classes = []

# Nouvelle fonction pour réinitialiser et revenir à la saisie de prompt
def new_prompt():
    st.session_state.user_prompt = ""
    st.session_state.user_prompt_english = ""
    st.session_state.prompt_submitted = False
    st.session_state.prompt_confirmed = False
    st.session_state.enhance_mode = False
    st.session_state.current_view = 'prompt'
    # Réinitialiser les mots-clés extraits
    st.session_state.keywords_extracted = False
    st.session_state.extracted_keywords = []
    # Réinitialiser les classifications
    st.session_state.keywords_classified = False
    st.session_state.objets_existants = []
    st.session_state.objets_inexistants = []
    st.session_state.sous_classes = []
          
    # AJOUTER LA RÉINITIALISATION DES SÉLECTIONS SPÉCIFIQUES
    st.session_state.selected_for_segmentation = []
    st.session_state.selected_for_finetuning = []
    st.session_state.selection_confirmed = False

    st.session_state.uploaded_image = None
    st.session_state.segmented_results = {}
    st.session_state.show_upload = False
    st.session_state.saved_image_path = None  # NOUVEAU: Réinitialiser le chemin sauvegardé
    st.session_state.segmentation_completed = False
    
    

    

# Mise à jour de la valeur de la prompt
def update_prompt(value):
    st.session_state.user_prompt = value
    st.session_state.prompt_error = False

# Fonction pour extraire les mots-clés en utilisant le LLM
def extract_keywords():
    # Utiliser la prompt confirmée (en anglais)
    prompt_to_process = st.session_state.user_prompt_english
 

    with st.spinner("🔍 Extraction des mots-clés en cours..."):
        start_time = time.time()
        # Utiliser la fonction fournie pour l'extraction des mots-clés
        extracted_objects = segment_objects_with_prompting(
            prompt_to_process,
            base_model_name="HuggingFaceH4/zephyr-7b-beta",
            lora_path="/teamspace/studios/this_studio/llm",
            max_new_tokens=128
        )
        end_time = time.time()

    # Mettre à jour les variables d'état
    st.session_state.extracted_keywords = extracted_objects
    st.session_state.keywords_extracted = True
    st.session_state.extraction_time = f"{end_time - start_time:.2f}"

    # Comparer avec les classes existantes
    classify_keywords()

    return extracted_objects


  
# Fonction pour classifier les mots-clés extraits
def classify_keywords():
    if st.session_state.extracted_keywords:
        json_path = "/teamspace/studios/this_studio/classeadapter.json"
        with st.spinner("🔍 Classification des mots-clés en cours..."):
            objets_ok, objets_nok, sous_classes_list = compare_keywords_with_class( # Renommé pour éviter conflit de nom
                st.session_state.extracted_keywords,
                json_path
            )

        # Mettre à jour les variables d'état avec les résultats
        st.session_state.objets_existants = objets_ok
        st.session_state.objets_inexistants = objets_nok
        st.session_state.sous_classes = sous_classes_list # Utilisation de la variable renommée
        st.session_state.keywords_classified = True



# ### REMPLACER L'ANCIENNE FONCTION PAR CELLE-CI ###

def start_segmentation():
    # Réinitialiser les résultats précédents avant de commencer
    st.session_state.segmented_results = {}
    seg_folder = "seg"
    os.makedirs(seg_folder, exist_ok=True)
    clear_folder(seg_folder) # Optionnel : vider les anciens résultats

    # Vérifier que toutes les conditions sont remplies
    if not st.session_state.get('saved_images_paths'):
        st.error("Veuillez d'abord importer au moins une image.")
        return
    if not st.session_state.get('selected_for_segmentation'):
        st.error("Veuillez sélectionner au moins un objet pour la segmentation.")
        return

    images_to_process = st.session_state.saved_images_paths
    total_images = len(images_to_process)
    progress_bar = st.progress(0)
    
    # Message global pour l'ensemble du processus
    status_text = st.empty()
    status_text.info(f"🧠 Démarrage de la segmentation pour {total_images} image(s)...")

    try:
        sam_checkpoint_path = "sam_vit_h_4b8939.pth"
        
        # Boucler sur chaque image importée
        for i, image_path in enumerate(images_to_process):
            image_filename = os.path.basename(image_path)
            
            # Mettre à jour le statut pour l'image en cours
            status_text.info(f"🧠 Traitement de l'image {i+1}/{total_images} : '{image_filename}'...")

            result_image = highlight_objects_in_image(
                img_path=image_path,
                keywords=st.session_state.selected_for_segmentation,
                sam_checkpoint_path=sam_checkpoint_path,
                show=False,
                threshold=0.1
            )

            if result_image is not None:
                if not isinstance(result_image, Image.Image):
                    result_image = Image.fromarray(result_image)

                # Sauvegarder l'image segmentée dans le dossier 'seg'
                output_path = os.path.join(seg_folder, f"segmented_{image_filename}")
                result_image.save(output_path)
                
                # Stocker le chemin du résultat dans le dictionnaire
                st.session_state.segmented_results[image_path] = output_path
            else:
                # Si aucun objet n'est trouvé, on stocke None pour le savoir
                st.session_state.segmented_results[image_path] = None
                st.warning(f"Aucun objet détecté pour l'image '{image_filename}'.")
            
            # Mettre à jour la barre de progression
            progress_bar.progress((i + 1) / total_images)

        status_text.success(f"✅ Segmentation terminée pour les {total_images} images !")
        st.session_state.segmentation_completed = True

    except FileNotFoundError as fnf_error:
        status_text.error(f"Erreur de fichier : {fnf_error}. Assurez-vous que le checkpoint SAM est présent.")
    except Exception as e:
        status_text.error(f"Une erreur est survenue durant la segmentation : {e}")
        import traceback
        st.code(traceback.format_exc())


# --- AJOUTER CES FONCTIONS À VOTRE SCRIPT ---

# Note : vous avez déjà normalize_coordinates et create_yolo_label,
# donc nous ajoutons les nouvelles fonctions nécessaires.

def draw_bounding_boxes_on_image(image, bboxes_info, colors=None):
    """Dessine les boîtes englobantes sur une copie de l'image."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    if colors is None:
        colors = ['#FF4136', '#3D9970', '#0074D9', '#FFDC00', '#B10DC9', '#FF851B']
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        
    for i, info in enumerate(bboxes_info):
        bbox = info['bbox']
        class_name = info['class']
        color = colors[i % len(colors)]
        
        draw.rectangle(bbox, outline=color, width=3)
        text = f"{i+1}: {class_name}"
        text_bbox = draw.textbbox((bbox[0], bbox[1]), text, font=font)
        # S'assurer que le fond du texte ne sort pas de l'image en haut
        text_y = bbox[1] if bbox[1] > text_bbox[3] - text_bbox[1] else bbox[1] + (text_bbox[3] - text_bbox[1])
        draw.rectangle((bbox[0], text_y - (text_bbox[3] - text_bbox[1]), bbox[0] + (text_bbox[2] - text_bbox[0]), text_y), fill=color)
        draw.text((bbox[0], text_y - (text_bbox[3] - text_bbox[1])), text, fill='white', font=font)
        
    return img_copy

def generate_finetuning_dataset():
    """Crée les dossiers et fichiers pour le dataset de fine-tuning dans 'data_fin'."""
    output_dir = "data_fin"
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    # Nettoyer et recréer les dossiers de destination pour un résultat propre
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Réinitialiser le mapping des classes pour qu'il soit propre à cette génération
    st.session_state.finetuning_class_mapping = {}
    
    # Parcourir uniquement les images qui ont des labels
    for img_path, labels_for_image in st.session_state.finetuning_all_labels.items():
        if not labels_for_image:
            continue # Ne pas inclure les images non labellisées

        base_filename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(base_filename)[0]

        # 1. Copier l'image dans data_fin/images
        shutil.copy(img_path, os.path.join(images_dir, base_filename))

        # 2. Créer le fichier .txt de labels dans data_fin/labels
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        yolo_label_strings = [
            create_yolo_label(
                label_info['class'],
                label_info['bbox'],
                img_width,
                img_height,
                st.session_state.finetuning_class_mapping
            ) for label_info in labels_for_image
        ]
        
        label_filepath = os.path.join(labels_dir, f"{filename_no_ext}.txt")
        with open(label_filepath, 'w') as f:
            f.write("\n".join(yolo_label_strings))

    # 3. Créer le fichier data.yaml, standard pour YOLO
    yaml_path = os.path.join(output_dir, "data.yaml")
    class_names = list(st.session_state.finetuning_class_mapping.keys())
    
    with open(yaml_path, 'w') as f:
        f.write(f"train: ../{images_dir}\n")
        f.write(f"val: ../{images_dir}\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    st.session_state.dataset_generated = True
    st.success(f"Dataset généré avec succès dans le dossier '{output_dir}' !")


def create_yolo_label(class_name, bbox, img_width, img_height, class_mapping):
    """
    Crée une chaîne de caractères de label au format YOLO à partir des informations d'une boîte.

    Args:
        class_name (str): Le nom de la classe de l'objet (ex: 'person').
        bbox (list): Les coordonnées [x1, y1, x2, y2] de la boîte.
        img_width (int): La largeur totale de l'image.
        img_height (int): La hauteur totale de l'image.
        class_mapping (dict): Un dictionnaire qui mappe les noms de classe à des ID numériques.
                              S'il est mis à jour, la modification est conservée.

    Returns:
        str: Le label au format YOLO "class_id x_center y_center width height".
    """
    # 1. Obtenir l'ID de la classe. Si la classe est nouvelle, l'ajouter au mapping.
    if class_name not in class_mapping:
        # L'ID sera la taille actuelle du dictionnaire (0, puis 1, etc.)
        new_id = len(class_mapping)
        class_mapping[class_name] = new_id
    class_id = class_mapping[class_name]

    # 2. Extraire les coordonnées de la boîte
    x1, y1, x2, y2 = bbox

    # 3. Calculer les dimensions et le centre de la boîte en pixels
    box_width = float(x2 - x1)
    box_height = float(y2 - y1)
    x_center = float(x1 + box_width / 2)
    y_center = float(y1 + box_height / 2)

    # 4. Normaliser toutes les valeurs par rapport aux dimensions de l'image (pour qu'elles soient entre 0 et 1)
    norm_x_center = x_center / img_width
    norm_y_center = y_center / img_height
    norm_width = box_width / img_width
    norm_height = box_height / img_height

    # 5. Formater la chaîne de caractères finale
    return f"{class_id} {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


# Fonctions associées aux boutons

def fine_tuning():
    st.warning("La fonctionnalité de Fine-Tuning est en cours de développement.")
    if not st.session_state.get('selected_for_finetuning'):
        st.error("Veuillez d'abord sélectionner des objets pour le Fine-Tuning.")
# ### AJOUTER CETTE NOUVELLE FONCTION ###
def clear_folder(folder_path):
    """Supprime tout le contenu d'un dossier et le recrée."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def zip_folder_to_bytes(folder_path):
    """Compresse un dossier en un fichier ZIP en mémoire et retourne les bytes."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # os.walk parcourt tous les fichiers et sous-dossiers
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Crée le chemin complet du fichier
                file_path = os.path.join(root, file)
                # Crée le nom du fichier dans l'archive (sans le chemin du dossier parent)
                archive_name = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname=archive_name)
                
    # Se positionne au début du buffer avant de le lire
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ----------- STREAMLIT UI --------------



st.markdown("""
    <style>
        /* Couleur de fond de la sidebar */
        section[data-testid="stSidebar"] {
            background-color: #FFEB99 !important;  /* gris clair */
        }
        /* Style des boutons dans la sidebar */
        section[data-testid="stSidebar"] button {
            background-color: #90EE90 !important; /* bleu clair */
            color: black !important;
            border: none !important;
            border-radius: 5px !important;
            padding: 0.5em 1em !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)
# Charger l'image
image = Image.open("SEG.png")

# Afficher l'image
st.image(image,width=700)

# Title of the app
st.title("SegmaVision Pro Light")

# Sidebar buttons to display different sections
st.sidebar.button("À propos", on_click=show_about,type="primary")
# Mise en évidence du bouton de retour pour nouvelle prompt
st.sidebar.button("Your Prompt", on_click=show_prompt, help="Cliquez ici pour entrer une nouvelle prompt",type="primary")
st.sidebar.button("Enhancer/Extraction", on_click=show_enhancer,type="primary") # MODIFIÉ ICI: Bouton Enhancer/Extraction dans la barre latérale


# Bouton pour afficher/cacher le file_uploader
if st.sidebar.button("Importer des données", key="import_data"):
    # 1. Vider le dossier et réinitialiser les états
    clear_folder("images_input")
    st.session_state.saved_images_paths = []
    st.session_state.segmented_results = {} 
    st.session_state.active_image_index = 0
    # Afficher l'uploader
    st.session_state.segmentation_completed = False
    st.session_state.show_upload = True
    

if st.session_state.get("show_upload", False):
    uploaded_files = st.sidebar.file_uploader(
        "Choisissez une ou plusieurs images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        input_folder = "images_input"
        saved_paths = []
        
        # 2. Enregistrer les nouvelles images
        for uploaded_file in uploaded_files:
            image_path = os.path.join(input_folder, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(image_path)
        
        st.session_state.saved_images_paths = saved_paths
        st.session_state.show_upload = False
        
        st.sidebar.success(f"{len(uploaded_files)} image(s) importée(s) !")
        st.rerun()

# ### AJOUTER CE BLOC POUR AFFICHER LES IMAGES SUR LA PAGE PRINCIPALE ###
# ### AJOUTER CE BLOC POUR AFFICHER LES IMAGES SUR LA PAGE PRINCIPALE ###
if st.session_state.saved_images_paths:
    st.markdown("---")
    st.subheader("🖼️ Images à traiter")

    # S'il y a plusieurs images, afficher un sélecteur
    if len(st.session_state.saved_images_paths) > 1:
        filenames = [os.path.basename(p) for p in st.session_state.saved_images_paths]
        selected_filename = st.selectbox(
            "Sélectionnez une image à analyser :",
            filenames,
            index=st.session_state.active_image_index
        )
        st.session_state.active_image_index = filenames.index(selected_filename)

    # Afficher l'image active et son résultat de segmentation si disponible
    active_path = st.session_state.saved_images_paths[st.session_state.active_image_index]
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(active_path, caption="Image originale", use_container_width=True) # MODIFIÉ ICI
    with col2:
        # ### MODIFIÉ ### : Logique pour chercher le résultat dans le dictionnaire
        # On vérifie si un résultat existe pour l'image active
        if active_path in st.session_state.segmented_results:
            segmented_image_path = st.session_state.segmented_results[active_path]
            
            # On vérifie si la segmentation a réussi pour cette image
            if segmented_image_path is not None:
                st.image(segmented_image_path, caption="Résultat de la segmentation", use_container_width=True)
            else:
                st.warning("Aucun objet n'a été détecté sur cette image lors du traitement.")
        else:
            st.info("Le résultat de la segmentation apparaîtra ici une fois le traitement lancé.")
    st.markdown("---")




# Afficher le contenu en fonction de l'état actuel
if st.session_state.current_view == 'about':
    # Platform definition
    st.subheader("Qu'est-ce que SegmaVision Pro Light ?")
    st.write("""
Bienvenue sur Segma Vision Pro Light, une plateforme puissante dédiée à la segmentation et à la labellisation automatique des images.
""")
    st.write("""
Conçue pour offrir précision, rapidité et évolutivité, Segma Vision Pro Light s’adapte à de nombreux domaines d’application tels que la vision par ordinateur, l'entraînement de modèles d’intelligence artificielle, et bien plus encore.
""")
    st.write("""
Cette solution innovante a été développée par BOURKIBA Salma et BOUTARHAT Alae, élèves ingénieurs en Génie Industriel, option Intelligence Artificielle et Data Science.
   
""")
    st.write("""
Vous trouverez ci-dessous un guide expliquant comment utiliser notre plateforme.
""")


    st.write("""
    1. Cliquez sur le bouton "Your Prompt" et saisissez une description : indiquez ce que vous souhaitez segmenter ou labelliser.
    2. Vous devez choisir ce que vous souhaitez, puis décider si vous souhaitez continuer avec votre instruction telle quelle — si vous êtes convaincu qu’elle est claire et correcte — ou bien l’améliorer davantage afin d’en augmenter la précision. 
    3. Si vous choisissez d'améliorer davantage votre instruction, vous devez cliquer sur le bouton "Enhancer" situé à gauche, afin de générer des versions correctes et claires de votre instruction.
    3. Sélection de la meilleure prompt : Choisissez la description qui correspond le mieux à vos besoins parmi plusieurs options proposées.
    4. Cliquez sur le bouton "Extraction des mots-clés" en bas.Extraction des mots-clés : Le système extrait automatiquement les objets à segmenter à partir de votre description.
    5. Démarrer le processus d'extraction des objets : Une fois les mots-clés identifiés, lancez le processus.
    6. Démarrer le processus de la segmentation : Lancement de l'algorithme de segmentation.
    7. Validation des résultats : Vérifiez si les objets segmentés sont corrects. Si nécessaire, ajustez les paramètres.
    """)

    st.subheader("""
    Contactez-nous:
""")


    st.write("""
📍 Location: Meknes, Morocco  
📧 Email: segmavision.team@gmail.com   
🔗 LinkedIn: linkedin.com/in/salma-bourkiba
             linkedin.com/in/alae-boutarhat
    """)



elif st.session_state.current_view == 'enhancer':
    st.subheader("🧠 Amélioration de prompt")

    st.write("Prompt initiale :")
    st.write(f"{st.session_state.user_prompt}")

    if st.session_state.enhanced_prompts:
        st.success(f"✅ Reformulations générées en {st.session_state.enhance_time} sec")
        st.info("Veuillez sélectionner une reformulation qui correspond le mieux à votre besoin :")

        # Utilisation de radio buttons pour la sélection d'une seule option
        options = []
        for i, (english_text, french_text, score) in enumerate(st.session_state.enhanced_prompts):
            options.append(f"{i+1}. {english_text}\n\n🇫🇷 : {french_text}")

        selected_option = st.radio(
            "Choisissez une reformulation :",
            options,
            index=st.session_state.selected_prompt_index if st.session_state.selected_prompt_index >= 0 else 0
        )

        # Mettre à jour l'index sélectionné
        selected_index = options.index(selected_option)
        select_enhanced_prompt(selected_index)

        # Afficher les détails de la reformulation sélectionnée
        if st.session_state.selected_prompt_index >= 0:
            english_text, french_text, score = st.session_state.enhanced_prompts[st.session_state.selected_prompt_index]

            st.write("### Reformulation sélectionnée :")
            st.write(f"*Original (EN)* : {english_text}")
            st.write(f"*Traduction (FR)* : {french_text}")

            # Bouton pour confirmer la sélection
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Utiliser cette reformulation", type="primary"):
                    confirm_selection()

        # Option pour revenir à la prompt initiale
        if st.button("Revenir à ma prompt initiale",type="primary"):
            st.session_state.current_view = 'prompt'

        # Option pour revenir au départ avec un message plus explicite
        if st.button("Je ne suis pas satisfait, nouvelle prompt", key="new_prompt_enhancer",type="primary"):
            new_prompt()
    else:
        st.warning("⚠ Aucune reformulation n'a pu être générée. Veuillez réessayer.")

        # Message plus explicite pour encourager l'utilisateur à recommencer
        st.error("""
        😕 Les reformulations n'ont pas pu être générées. Nous vous recommandons de retourner
        à l'étape de saisie et d'entrer une nouvelle prompt plus détaillée.
        """)

        # Bouton pour entrer une nouvelle prompt
        if st.button("Entrer une nouvelle prompt", key="new_prompt_no_results"):
            new_prompt()

elif st.session_state.current_view == 'prompt':
    st.subheader("Saisir la description de l'objet à segmenter ou labelliser")

    # Si la prompt n'est pas encore confirmée
    if not st.session_state.prompt_confirmed:
        # Si la prompt n'est pas encore soumise, afficher la zone de texte
        if not st.session_state.prompt_submitted:
            # Utiliser une clé unique pour le text_area
            prompt_input = st.text_area(
                "Décrivez votre besoin ici:",
                value=st.session_state.user_prompt,
                height=150,
                key="prompt_input"
            )

            # Mettre à jour st.session_state.user_prompt
            if prompt_input != st.session_state.user_prompt:
                st.session_state.user_prompt = prompt_input

            # Create a column layout: input on the left, button on the right
            col1, col2 = st.columns([3, 1])

            with col2:
                # Button to confirm the prompt entry
                st.button("Entrée", on_click=submit_prompt)

            # Afficher le message d'erreur si nécessaire
            if 'prompt_error' in st.session_state and st.session_state.prompt_error:
                st.error("Veuillez saisir une description avant de cliquer sur 'Entrée'.")

        # Si la prompt est soumise mais pas encore confirmée
        if st.session_state.prompt_submitted and not st.session_state.prompt_confirmed:
            st.success("Prompt soumise avec succès!")
            st.write("Prompt saisie :")
            st.write(st.session_state.user_prompt)

            # Indiquer si la prompt est en français et sera traduite
            if est_en_francais(st.session_state.user_prompt):
                st.info("Votre prompt est en français et sera automatiquement traduite en anglais si vous continuez sans amélioration.")

            # Demander à l'utilisateur s'il souhaite continuer avec cette prompt ou l'améliorer
            st.write("### Souhaitez-vous continuer avec cette prompt ou l'améliorer ?")

            col1, col2 = st.columns(2)
            with col1:
                st.button("Continuer avec cette prompt", on_click=confirm_prompt, type="primary")

                

            # Information pour entrer une nouvelle prompt
            st.info("💡 Si vous souhaitez modifier complètement votre prompt, cliquez sur le bouton ci-dessous pour recommencer.")
            st.info(f"""
                Pour améliorer votre prompt, veuillez cliquer sur le bouton 'Enhancer/Extraction' qui se trouve dans la barre latérale gauche.
                """)
            if st.button("Entrer une nouvelle prompt", on_click=new_prompt):
                pass

          

    # Si la prompt est confirmée, passer à l'étape suivante
    if st.session_state.prompt_confirmed:
        st.success("Prompt confirmée!")

        # Afficher la prompt originale et sa traduction si elle est différente
        if hasattr(st.session_state, 'user_prompt_original') and st.session_state.user_prompt_original != st.session_state.user_prompt_english:
            st.write("Prompt originale :")
            st.write(f"{st.session_state.user_prompt_original}")
            st.write("Prompt traduite pour le traitement :")
            st.write(f"{st.session_state.user_prompt_english}")
        else:
            st.write("Prompt pour le traitement :")
            st.write(f"{st.session_state.user_prompt_english}")

        # Bouton pour extraire les mots-clés une fois la prompt confirmée
        if not st.session_state.keywords_extracted:
            st.info("""
            ⚠️ **Pour extraire les classes à segmenter ou à labelliser**, veuillez cliquer sur le bouton
            "Extraire les mots-clés" ci-dessous. 
            """)

            if st.button("Extraire les mots-clés", key="extract_confirmed_prompt"):
                extract_keywords()

        # Afficher les mots-clés extraits si disponibles
        if st.session_state.keywords_extracted:
            st.success(f"✅ Extraction des mots-clés réalisée en {st.session_state.extraction_time} sec")

            if st.session_state.extracted_keywords:
                st.write("### Objets à segmenter :")

                # Afficher les mots-clés classifiés
                # LE NOUVEAU BLOC DE CODE AVEC SÉLECTION
                # =================== BLOC DE SÉLECTION FINAL ET ROBUSTE ===================
                if st.session_state.keywords_classified:
                    if not any([st.session_state.objets_existants, st.session_state.objets_inexistants, st.session_state.sous_classes]):
                        st.warning("⚠️ Aucun objet à segmenter n'a été détecté. Veuillez essayer une autre prompt.")
                    else:
                        

                        if not st.session_state.selection_confirmed:
                            st.info("Veuillez cocher les objets que vous souhaitez traiter (pour la segmentation ou le fine-tuning).")

                            # Listes temporaires pour capturer l'état actuel des checkboxes
                            current_selection_seg = []
                            current_selection_ft = []

                            col1, col2, col3 = st.columns(3)

                            # --- Colonne 1: Objets Existants (pour Segmentation) ---
                            with col1:
                                st.write("#### ✅ Pour Segmentation")
                                if st.session_state.objets_existants:
                                    for obj in st.session_state.objets_existants:
                                        # La valeur de la checkbox est True si l'objet est déjà dans la liste de sélection
                                        is_checked = st.checkbox(
                                            obj, 
                                            key=f"cb_seg_{obj}",
                                            value=(obj in st.session_state.selected_for_segmentation)
                                        )
                                        if is_checked:
                                            current_selection_seg.append(obj)
                                else:
                                    st.write("Aucun")
                            
                            # Mettre à jour la liste principale avec la sélection actuelle
                            st.session_state.selected_for_segmentation = current_selection_seg

                            # --- Colonne 2 & 3 : Objets pour Fine-Tuning ---
                            with col2:
                                st.write("#### 🔧 Pour Fine-Tuning")
                                if st.session_state.objets_inexistants:
                                    st.write("**Nouveaux Objets :**")
                                    for obj in st.session_state.objets_inexistants:
                                        is_checked = st.checkbox(
                                            obj, 
                                            key=f"cb_ft_inex_{obj}",
                                            value=(obj in st.session_state.selected_for_finetuning)
                                        )
                                        if is_checked:
                                            current_selection_ft.append(obj)
                                else:
                                    st.write("_Aucun nouvel objet_")

                            with col3:
                                st.write("####  ") # Espace pour aligner le titre
                                if st.session_state.sous_classes:
                                    st.write("**Nouvelles Sous-Classes :**")
                                    for obj in st.session_state.sous_classes:
                                        is_checked = st.checkbox(
                                            obj, 
                                            key=f"cb_ft_sous_{obj}",
                                            value=(obj in st.session_state.selected_for_finetuning)
                                        )
                                        if is_checked:
                                            # Vérifier que l'on n'ajoute pas de doublons si un objet est dans les deux listes
                                            if obj not in current_selection_ft:
                                                current_selection_ft.append(obj)
                                else:
                                    st.write("_Aucune nouvelle sous-classe_")
                            
                            # Mettre à jour la liste principale avec la sélection actuelle
                            st.session_state.selected_for_finetuning = current_selection_ft


                            st.markdown("---")

                            # Activer le bouton si au moins une sélection a été faite
                            if st.session_state.selected_for_segmentation or st.session_state.selected_for_finetuning:
                                if st.button("Confirmer la sélection", type="primary"):
                                    st.session_state.selection_confirmed = True
                                    st.rerun()
                            else:
                                st.warning("Veuillez sélectionner au moins un objet pour continuer.")

                        # --- Affichage après confirmation (pas de changement ici) ---
                        if st.session_state.selection_confirmed:
                            st.success("Sélection confirmée !")
                            
                            
                            if st.session_state.selected_for_segmentation:
                                st.markdown("#### Objets prêts pour la **Segmentation** :")
                                for obj in st.session_state.selected_for_segmentation:
                                    st.markdown(f"- ✅ **{obj}**")
                                
                            
                            if st.session_state.selected_for_finetuning:
                                st.markdown("#### Objets nécessitant un **Fine-Tuning** :")
                                for obj in st.session_state.selected_for_finetuning:
                                    st.markdown(f"- 🔧 **{obj}**")
                                

                            if not st.session_state.selected_for_segmentation and not st.session_state.selected_for_finetuning:
                                st.warning("Aucun objet n'a été sélectionné. Veuillez recommencer le processus.")



                                 

                st.info("""     

Bravo, vous êtes arrivé à ce stade ! Voici maintenant comment cela fonctionne :

Il existe généralement trois types de sorties de classes :

Les classes existantes : Ce sont les classes que l’on connaît déjà. On peut donc segmenter automatiquement les données qui leur sont associées.
👉 Pour cela, cliquez sur le bouton "Démarrer la segmentation".

Les classes inexistantes : Ce sont des classes inconnues du système. Elles ne peuvent pas être segmentées automatiquement. Il est donc nécessaire d’ajouter de nouvelles données et de les annoter manuellement pour que le modèle puisse les apprendre. Une fois cela fait, elles pourront être segmentées comme les autres.
👉 Pour cela, cliquez sur le bouton "Fine Tuning".

Les sous-classes : Ici, seules les classes générales sont connues, sans distinction précise des sous-catégories. Il faut donc ajouter des données supplémentaires afin que le modèle puisse apprendre à les différencier.
👉 Là aussi, cliquez sur le bouton "Fine Tuning".



                """)
            else:
                st.warning("⚠️ Aucun objet à segmenter n'a été détecté. Veuillez essayer une autre prompt.")





elif st.session_state.current_view == 'finetuning':
    st.header("🛠️ Interface de Labellisation pour le Fine-Tuning")
    st.markdown("Pour les classes nouvelles ou à préciser, veuillez dessiner des boîtes englobantes en entrant leurs coordonnées.")

    if not st.session_state.saved_images_paths:
        st.error("Aucune image à labelliser. Veuillez retourner importer des données via la barre latérale.")
    else:
        col_main, col_sidebar = st.columns([2, 1])

        # --- COLONNE DE DROITE : LES CONTRÔLES ---
        with col_sidebar:
            st.subheader("⚙️ Panneau de Contrôle")

            # 1. Sélectionner l'image active
            # (Vérifier si finetuning_active_image_path est initialisé)
            if st.session_state.finetuning_active_image_path is None:
                 st.session_state.finetuning_active_image_path = st.session_state.saved_images_paths[0]
            
            image_filenames = [os.path.basename(p) for p in st.session_state.saved_images_paths]
            selected_filename = st.selectbox(
                "1. Choisissez une image à annoter :",
                image_filenames,
                index=image_filenames.index(os.path.basename(st.session_state.finetuning_active_image_path))
            )
            st.session_state.finetuning_active_image_path = st.session_state.saved_images_paths[image_filenames.index(selected_filename)]
            
            active_image_path = st.session_state.finetuning_active_image_path
            image_to_label = Image.open(active_image_path)
            img_width, img_height = image_to_label.size
            st.info(f"Dimensions de l'image : {img_width}x{img_height} pixels")

            # 2. Saisir une nouvelle boîte
            with st.form("new_bbox_form", clear_on_submit=True):
                st.markdown("**2. Ajoutez une nouvelle boîte :**")
                
                selected_class = st.selectbox(
                    "Classe de l'objet :",
                    st.session_state.selected_for_finetuning
                )
                
                c1, c2 = st.columns(2)
                x1 = c1.number_input("X1 (gauche)", min_value=0, max_value=img_width, step=10, key="x1")
                y1 = c1.number_input("Y1 (haut)", min_value=0, max_value=img_height, step=10, key="y1")
                x2 = c2.number_input("X2 (droite)", min_value=0, max_value=img_width, step=10, key="x2")
                y2 = c2.number_input("Y2 (bas)", min_value=0, max_value=img_height, step=10, key="y2")
                
                submitted = st.form_submit_button("➕ Ajouter cette boîte", type="primary")

                if submitted:
                    if selected_class and x2 > x1 and y2 > y1:
                        new_label = {'class': selected_class, 'bbox': [int(x1), int(y1), int(x2), int(y2)]}
                        if active_image_path not in st.session_state.finetuning_all_labels:
                            st.session_state.finetuning_all_labels[active_image_path] = []
                        st.session_state.finetuning_all_labels[active_image_path].append(new_label)
                        st.success(f"Boîte ajoutée pour '{selected_class}' !")
                        st.rerun()
                    else:
                        st.error("Coordonnées invalides (X2 doit être > X1 et Y2 > Y1).")

            # 3. Gérer les labels existants
            st.markdown("**Labels sur cette image :**")
            labels_on_current_image = st.session_state.finetuning_all_labels.get(active_image_path, [])
            if labels_on_current_image:
                for i, label in enumerate(labels_on_current_image):
                    st.text(f"  {i+1}. {label['class']} : {label['bbox']}")
                if st.button("🗑️ Supprimer le dernier label", use_container_width=True):
                    st.session_state.finetuning_all_labels[active_image_path].pop()
                    st.rerun()
            else:
                st.write("_Aucun label pour le moment._")

            # 4. Finaliser
            st.subheader("🎯 Finalisation")
            if st.button("✅ Terminer et Générer le Dataset", use_container_width=True):
                with st.spinner("Création du dataset dans le dossier 'data_fin'..."):
                    generate_finetuning_dataset()
                st.rerun()

            if st.session_state.dataset_generated:
                st.info("Le dataset a été généré localement dans le dossier `data_fin`.")
                try:
                    zip_data = zip_folder_to_bytes("data_fin")
                    st.download_button(
                        label="📥 Télécharger le Dataset (ZIP)",
                        data=zip_data,
                        file_name="finetuning_dataset.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erreur lors de la création du ZIP: {e}")

        # --- COLONNE DE GAUCHE : LA VISUALISATION ---
        with col_main:
            st.subheader("🖼️ Visualisation de l'Annotation")
            labels_to_draw = st.session_state.finetuning_all_labels.get(active_image_path, [])
            
            if labels_to_draw:
                img_with_boxes = draw_bounding_boxes_on_image(image_to_label, labels_to_draw)
                st.image(img_with_boxes, use_container_width=True, caption="Image avec les boîtes labellisées")
            else:
                st.image(image_to_label, use_container_width=True, caption="Image originale - Prête pour l'annotation")
# ---- FIN DU BLOC AJOUTÉ ----



# ➕ Bouton pour démarrer la segmentation si les conditions sont remplies
if st.sidebar.button("Démarrer la segmentation", key="start_segmentation_sidebar"):
    if not st.session_state.get('saved_images_paths'):
        st.sidebar.warning("Veuillez d'abord importer une image.")
    elif not st.session_state.get('selected_for_segmentation'):
        st.sidebar.warning("Veuillez sélectionner au moins un objet à segmenter.")
    else:
        start_segmentation()


# ... après le bloc if st.sidebar.button("Démarrer la segmentation" ...

# ➕ Bouton pour télécharger les résultats si la segmentation est terminée
if st.session_state.get("segmentation_completed", False):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Télécharger les résultats")

    # Compresser le dossier 'seg' en bytes
    try:
        zip_data = zip_folder_to_bytes("seg")
        st.sidebar.download_button(
            label="📥 Télécharger seg.zip",
            data=zip_data,
            file_name="seg.zip",
            mime="application/zip",
            type="primary",
            help="Cliquez pour télécharger toutes les images segmentées dans un fichier ZIP."
        )
    except Exception as e:
        st.sidebar.error(f"Erreur lors de la création du ZIP: {e}")




# --- AJOUTER CE BLOC DANS LA SIDEBAR (après le bouton Segmentation) ---


# Ce bouton est conditionnel : il n'est utile que si des objets sont sélectionnés pour le fine-tuning
if st.session_state.get('selected_for_finetuning'):
    if st.sidebar.button("🛠️ Démarrer le Fine-Tuning", key="start_finetuning_sidebar", type="primary"):
        if not st.session_state.get('saved_images_paths'):
            st.sidebar.warning("Veuillez d'abord importer des images.")
        else:
            # Préparer l'état pour la vue de fine-tuning
            st.session_state.finetuning_all_labels = {path: [] for path in st.session_state.saved_images_paths}
            st.session_state.finetuning_active_image_path = st.session_state.saved_images_paths[0]
            st.session_state.dataset_generated = False
            st.session_state.current_view = 'finetuning' # Changer de vue
            st.rerun()
















