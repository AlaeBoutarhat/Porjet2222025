
import streamlit as st
import json
import subprocess
import numpy as np
from bert_score import BERTScorer
import random
import time
from deep_translator import GoogleTranslator
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

####
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor


'''

def segment_objects_with_prompting(prompt,
                                    base_model_name="HuggingFaceH4/zephyr-7b-beta",
                                    lora_path="/teamspace/studios/this_studio/llm",
                                    max_new_tokens=128):
    system_instruction = (
    "You are a world-class object extraction expert for vision-language tasks. "
    "Your only goal is to extract all physical, visible objects mentioned or implied in a user’s prompt, "
    "to prepare for segmentation in an image.\n\n"

    "🧠 You understand both simple and complex prompts, even when the object mentions are indirect, implied, or embedded in long instructions.\n\n"

    "🔍 Your job is to:\n"
    "1. Identify every concrete, visible, segmentable object mentioned in the prompt.\n"
    "2. Return ONLY a **clean, comma-separated list** of these object names.\n\n"

    "📌 STRICT RULES:\n"
    "- ✅ Output only singular, normalized object names (e.g., 'Dog', not 'Dogs').\n"
    "- ✅ Capitalize each object (e.g., 'Tree', 'Car', 'Person').\n"
    "- ❌ Do NOT include colors, actions, verbs, adjectives, or scene descriptions.\n"
    "- ❌ Do NOT include background elements unless explicitly asked (e.g., 'Sky', 'Ground').\n"
    "- ❌ Do NOT repeat objects. No explanations. No formatting. Only the list.\n\n"

    "🧪 Examples:\n"
    "➡ Prompt: 'Segment dogs, cars, and any people, but ignore trees and the sky.'\n"
    "✔ Output: Dog, Car, Person\n\n"

    "➡ Prompt: 'Please segment everything related to food, like apples, bananas, or bread.'\n"
    "✔ Output: Apple, Banana, Bread\n\n"

    "➡ Prompt: 'I want to segment animals such as horses, birds, and cats. Skip buildings and humans.'\n"
    "✔ Output: Horse, Bird, Cat\n\n"

    "⛔ Bad Outputs:\n"
    "- 'Segmented objects: Dog, Car'\n"
    "- 'I found: Cat, Dog'\n"
    "- 'Apple, Banana, Bread. Ignore cups.'\n\n"

    "🔁 Always return a minimal and clean list like:\n"
    "👉 Dog, Car, Tree, Person\n\n"

    "🧠 Be comprehensive. Be precise. Only return valid object names."
)


    full_prompt = f"<|system|>\n{system_instruction}\n<|user|>\n{prompt}\n<|assistant|>\n"
    device = torch.device("cpu")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map={"": device}
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.to(device)
    model.eval()

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            num_return_sequences=1,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = result.split("<|assistant|>")[-1].strip()

    lines = [line.strip() for line in assistant_response.splitlines() if line.strip()]
    if lines:
        object_line = lines[0]
        object_candidates = [obj.strip().capitalize() for obj in object_line.split(',') if obj.strip()]
        cleaned_objects = list(set(object_candidates))
    else:
        cleaned_objects = []

    return cleaned_objects


'''







import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def segment_objects_with_prompting(prompt,
                                    base_model_name="HuggingFaceH4/zephyr-7b-beta",
                                    lora_path="/teamspace/studios/this_studio/llm",
                                    max_new_tokens=128):
    
    system_instruction = (
        "You are a world-class object extraction expert for vision-language tasks. "
        "Your only goal is to extract all physical, visible objects mentioned or implied in a user’s prompt, "
        "to prepare for segmentation in an image.\n\n"

        "🧠 You understand both simple and complex prompts, even when the object mentions are indirect, implied, or embedded in long instructions.\n\n"

        "🔍 Your job is to:\n"
        "1. Identify every concrete, visible, segmentable object mentioned in the prompt.\n"
        "2. Return ONLY a **clean, comma-separated list** of these object names.\n\n"

        "📌 STRICT RULES:\n"
        "- ✅ Output only singular, normalized object names (e.g., 'Dog', not 'Dogs').\n"
        "- ✅ Capitalize each object (e.g., 'Tree', 'Car', 'Person').\n"
        "- ❌ Do NOT include colors, actions, verbs, adjectives, or scene descriptions.\n"
        "- ❌ Do NOT include background elements unless explicitly asked (e.g., 'Sky', 'Ground').\n"
        "- ❌ Do NOT repeat objects. No explanations. No formatting. Only the list.\n"
        "- ❌ Do NOT return a full sentence. ONLY output object names in a clean list.\n\n"

        "🧪 Examples:\n"
        "➡ Prompt: 'Segment dogs, cars, and any people, but ignore trees and the sky.'\n"
        "✔ Output: Dog, Car, Person\n\n"

        "➡ Prompt: 'Please segment everything related to food, like apples, bananas, or bread.'\n"
        "✔ Output: Apple, Banana, Bread\n\n"

        "➡ Prompt: 'I want to segment animals such as horses, birds, and cats. Skip buildings and humans.'\n"
        "✔ Output: Horse, Bird, Cat\n\n"

        "⛔ Bad Outputs:\n"
        "- 'Segmented objects: Dog, Car'\n"
        "- 'I found: Cat, Dog'\n"
        "- 'Apple, Banana, Bread. Ignore cups.'\n"

        "🔁 Always return a minimal and clean list like:\n"
        "👉 Dog, Car, Tree, Person\n\n"

        "🧠 Be comprehensive. Be precise. Only return valid object names."
    )

    full_prompt = f"<|system|>\n{system_instruction}\n<|user|>\n{prompt}\n<|assistant|>\n"
    device = torch.device("cpu")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map={"": device}
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.to(device)
    model.eval()

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            num_return_sequences=1,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = result.split("<|assistant|>")[-1].strip()

    lines = [line.strip() for line in assistant_response.splitlines() if line.strip()]
    if lines:
        object_line = lines[0]
        object_candidates = [obj.strip().capitalize() for obj in object_line.split(',') if obj.strip()]
        cleaned_objects = list(set(object_candidates))
    else:
        cleaned_objects = []

    return cleaned_objects








def traduire_en_francais(phrase):
    try:
        traduction = GoogleTranslator(source='auto', target='fr').translate(phrase)
        return traduction
    except Exception as e:
        st.error(f"❌ Erreur lors de la traduction : {e}")
        return phrase  # En cas d'erreur, retourner la phrase originale

def traduire_en_anglais(phrase):
    try:
        traduction = GoogleTranslator(source='auto', target='en').translate(phrase)
        return traduction
    except Exception as e:
        st.error(f"❌ Erreur lors de la traduction : {e}")
        return phrase  # En cas d'erreur, retourner la phrase originale

def est_en_francais(texte):
    """Détecte si le texte est probablement en français"""
    try:
        # Utiliser la détection automatique de langue de GoogleTranslator
        langue = GoogleTranslator(source='auto', target='en').detect(texte)
        return langue.lower() == 'fr'
    except:
        # En cas d'erreur, on fait une détection basique basée sur les accents et mots français courants
        mots_francais = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'je', 'tu', 'il', 'elle', 'nous', 'vous']
        texte_lower = texte.lower()
        # Vérifier la présence d'accents et de mots français courants
        contient_accents = any(c in texte_lower for c in 'éèêëàâäôöùûüç')
        contient_mots_fr = any(f" {mot} " in f" {texte_lower} " for mot in mots_francais)
        return contient_accents or contient_mots_fr

    







def send_prompt_to_mistral(prompt_text, temperature=0.6):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt_text,
                "temperature": temperature,
                "stream": False
            }
        )
        response.raise_for_status()
        result = response.json()
        output = result.get("response", "")
        # Retourner la liste des reformulations, en nettoyant les lignes vides
        return [line.strip() for line in output.strip().split("\n") if line.strip()]
    except Exception as e:
        print(f"❌ Erreur API Mistral : {e}")
        return []


def get_best_reformulations(prompt, dataset_path="dt.txt", num_best=5, temperature=0.6):
    final_prompt = f"""
Tu es un assistant linguistique intelligent. Voici les étapes que tu dois suivre pour chaque prompt que je te donne :

1. Nettoyage : 
   - Supprime les caractères spéciaux non significatifs. ( par exemple (......)(////)(;';.;;')))
   - Transforme toute la phrase en minuscules.

2. Correction : 
   - Corrige toutes les fautes d'orthographe et grammaticales.

3. Traduction : 
   - Si la phrase est en français, traduis-la en anglais.
   - Si elle est déjà en anglais, ne fais pas de traduction.

4. Reformulations :
   - Génère 10 reformulations différentes en anglais, avec un sens strictement équivalent.
   - Les 10 reformulations ne doivent pas contenir de caractères spéciaux. 
   - Les 10 reformulations ne doivent pas contenir de répétitions.

Voici la prompt à traiter :
\"\"\"{prompt}\"\"\"
"""
    reformulations = send_prompt_to_mistral(final_prompt, temperature=temperature)
    if not reformulations:
        return []

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            all_lines = []
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "input" in data:
                        all_lines.append(data)
                except json.JSONDecodeError:
                    continue
            dataset = [entry["input"] for entry in random.sample(all_lines, min(10, len(all_lines)))]
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du dataset : {e}")
        return []

    if not dataset:
        st.warning("⚠ Dataset vide ou mal formaté.")
        return []

    # Nettoyer les reformulations (minimum 4 mots pour éviter reformulations trop courtes)
    reformulations = [r for r in reformulations if len(r.split()) > 3]
    if not reformulations:
        st.warning("⚠ Aucune reformulation utilisable.")
        return []

    # Calculer les scores BERTScore
    original_sentences = []
    generated_sentences = []
    for original in dataset:
        for reform in reformulations:
            original_sentences.append(original)
            generated_sentences.append(reform)

    try:
        scorer = BERTScorer(lang="en", model_type="roberta-base", rescale_with_baseline=True)
        P, R, F1 = scorer.score(generated_sentences, original_sentences)
    except Exception as e:
        st.error(f"❌ Erreur lors du calcul de BERTScore : {e}")
        return []

    # Trier les reformulations par score (F1 moyen sur tout le dataset)
    # On calcule la moyenne des scores pour chaque reformulation sur tous les inputs du dataset
    scores_per_reform = []
    n_dataset = len(dataset)
    for i in range(len(reformulations)):
        # F1 scores pour la reformulation i sont à l'indice i + k*len(reformulations) pour k in dataset
        indices = [i + k*len(reformulations) for k in range(n_dataset)]
        mean_score = np.mean([F1[idx].item() for idx in indices])
        scores_per_reform.append(mean_score)

    best_indices = np.argsort(scores_per_reform)[::-1][:num_best]

    results = []
    for i in best_indices:
        english_text = reformulations[i]
        french_text = traduire_en_francais(english_text)
        results.append((english_text, french_text, scores_per_reform[i]))

    return results














import json

def compare_keywords_with_class(keywords, json_class_path):
    # Charger la liste d'objets depuis le fichier JSON
    with open(json_class_path, 'r') as f:
        class_objects = json.load(f)

    objets_existants = []
    objets_inexistants = []
    sous_classes = []

    for keyword in keywords:
        keyword_lower = keyword.lower()
        found_match = False

        for class_obj in class_objects:
            class_obj_lower = class_obj.lower()

            # Comparaison lettre par lettre
            match = True
            for i in range(len(class_obj_lower)):
                if i >= len(keyword_lower) or keyword_lower[i] != class_obj_lower[i]:
                    match = False
                    break

            if match:
                if keyword_lower == class_obj_lower:
                    objets_existants.append(keyword)
                else:
                    sous_classes.append(keyword)
                found_match = True
                break  # On arrête dès qu’un match est trouvé

        if not found_match:
            objets_inexistants.append(keyword)

    return objets_existants, objets_inexistants, sous_classes







def highlight_objects_in_image(
    img_path,
    keywords,
    owlvit_model_name="google/owlvit-base-patch32",
    sam_checkpoint_path="sam_vit_h_4b8939.pth",
    show=True,
    threshold=0.005
):
    """
    Pipeline complet : OWL-ViT + SAM.
    img_path : chemin vers l'image (jpg/png).
    keywords : liste de chaînes (ex : ["dog", "cat"]).
    show : True -> affiche l'image, False -> retourne l'image annotée (numpy).
    threshold : seuil de détection pour OWL-ViT.
    """
    # Charger l'image
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image non trouvée : {img_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Charger OWL-ViT
    processor = OwlViTProcessor.from_pretrained(owlvit_model_name)
    model = OwlViTForObjectDetection.from_pretrained(owlvit_model_name)

    # Préparer la requête textuelle
    queries = [keywords]

    # Détection OWL-ViT
    inputs = processor(text=queries, images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image_rgb.shape[:2]])
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    if len(boxes) == 0:
        print("Aucun objet détecté, essayez un seuil plus bas ou d'autres mots-clés.")
        if show:
            plt.imshow(image_rgb)
            plt.title("Aucune détection")
            plt.axis('off')
            plt.show()
        return None

    # Charger SAM
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # Annotation
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_rgb)
    colors = ["lime", "cyan", "yellow", "magenta", "orange", "red", "blue"]
    # Pour chaque box détectée, appliquer SAM et dessiner le mask
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        box_np = box.cpu().numpy()
        mask, _, _ = predictor.predict(
            box=box_np,
            multimask_output=False
        )
        color = colors[i % len(colors)]
        ax.contour(mask[0], colors=color, linewidths=2)
        x0, y0, x1, y1 = box_np.astype(int)
        ax.add_patch(
            plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor=color, linewidth=2)
        )
        caption = f"{keywords[label]} ({score:.2f})"
        ax.text(x0, y0-5, caption, color=color, fontsize=12, weight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=1))

    ax.axis('off')
    ax.set_title("Objets détectés")
    plt.tight_layout()
    if show:
        plt.show()
        return None
    else:
        # Retourne l'image annotée (en RGB, numpy array)
        fig.canvas.draw()
        img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_arr
