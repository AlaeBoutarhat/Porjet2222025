import re

def nettoyer_prompt(prompt):
    # Supprimer les caractères spéciaux (mais garder lettres, chiffres et espaces)
    prompt_sans_speciaux = re.sub(r'[^\w\s]', '', prompt)
    
    # Mettre en minuscules
    prompt_nettoye = prompt_sans_speciaux.lower()
    
    return prompt_nettoye


import language_tool_python
'''
def corriger_phrase(phrase):
    tool = language_tool_python.LanguageTool('fr') 
    # 'fr' pour le français, 'en-US' pour l'anglais
    correction = tool.correct(phrase)
    return correction
import language_tool_python
'''
def corriger_phrase(phrase, langue='fr'):
    # Créez l'objet de vérification grammaticale en fonction de la langue
    if langue == 'fr':
        tool = language_tool_python.LanguageTool('fr')  # Pour le français
    elif langue == 'en':
        tool = language_tool_python.LanguageTool('en-US')  # Pour l'anglais
    else:
        raise ValueError("Langue non supportée. Utilisez 'fr' ou 'en'.")
    
    # Correction de la phrase
    correction = tool.correct(phrase)
    return correction

from deep_translator import GoogleTranslator

def traduire_en_anglais(phrase):
    traduction = GoogleTranslator(source='auto', target='en').translate(phrase)
    return traduction
def traduire_en_francais(phrase):
    traduction = GoogleTranslator(source='auto', target='fr').translate(phrase)
    return traduction

'''
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Charger le modèle spécialisé dans la paraphrase
tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")

def reformulate(prompt):
    input_text = f"Paraphrase cette phrase : {prompt}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        input_ids,
        max_length=80,  # Augmenter la longueur max pour plus de diversité
        num_return_sequences=10,  # Générer 5 reformulations différentes
        temperature=2.0,  # Augmenter la température pour plus de créativité
        top_k=50,         # Sélection des 50 mots les plus probables
        top_p=0.8,        # Probabilité cumulative (nucleus sampling)
        do_sample=True,   # Active l'échantillonnage pour utiliser temperature, top_k et top_p
        repetition_penalty=1.2,  # Pénaliser les répétitions pour encourager la diversité
        no_repeat_ngram_size=2   # Empêcher la répétition de n-grammes
    )
    
    # Utiliser un set pour garantir l'unicité des reformulations
    reformulations = set()
    for output in outputs:
        result = tokenizer.decode(output, skip_special_tokens=True)
        reformulations.add(result)  # Ajouter la reformulation au set

    # Retourner une liste unique de reformulations
    return list(reformulations)

# Test du modèle
prompt = "Segment dogs without considering cats."
results = reformulate(prompt)

# Afficher les reformulations générées
for i, result in enumerate(results, 1):
    print(f"✅ Reformulation {i} : {result}")'
'''




import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from bert_score import score
import numpy as np

# Charger le modèle spécialisé dans la paraphrase
tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")

# Fonction combinée reform
def reform(prompt, dataset_path, num_reformulations=10, num_best=6):
    # 1. Génération des reformulations
    input_text = f"Paraphrase cette phrase : {prompt}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        input_ids,
        max_length=80,  # Augmenter la longueur max pour plus de diversité
        num_return_sequences=num_reformulations,  # Générer plusieurs reformulations différentes
        temperature=1.8,  # Augmenter la température pour plus de créativité
        top_k=50,         # Sélection des 50 mots les plus probables
        top_p=0.8,        # Probabilité cumulative (nucleus sampling)
        do_sample=True,   # Active l'échantillonnage pour utiliser temperature, top_k et top_p
        repetition_penalty=1.2,  # Pénaliser les répétitions pour encourager la diversité
        no_repeat_ngram_size=2   # Empêcher la répétition de n-grammes
    )
    
    # Utiliser un set pour garantir l'unicité des reformulations
    reformulations = set()
    for output in outputs:
        result = tokenizer.decode(output, skip_special_tokens=True)
        reformulations.add(result)  # Ajouter la reformulation au set

    generated_reformulations = list(reformulations)

    # 2. Charger le dataset contenant les prompts
    dataset = []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())  # Vérifier que chaque ligne est un JSON valide
                    if "input" in data:
                        dataset.append(data["input"])  # Ajouter seulement la phrase originale
                except json.JSONDecodeError:
                    print(f"⚠️ Erreur de décodage JSON sur la ligne : {line.strip()}")
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé : {dataset_path}")
        return [], []

    # Si le dataset est vide ou mal formaté
    dataset_prompts = dataset[:10]
    if not dataset_prompts:
        print("⚠️ Le dataset est vide ou mal formaté.")
        return [], []

    original_sentences = []
    generated_sentences = []

    # Créer les listes de phrases originales et reformulées
    for original in dataset_prompts:
        for generated in generated_reformulations:
            original_sentences.append(original)
            generated_sentences.append(generated)

    # 3. Calculer le score BERTScore entre les reformulations générées et les phrases originales
    P, R, F1 = score(generated_sentences, original_sentences, model_type="roberta-large", lang="en")
    
    # Récupérer les indices des meilleures reformulations
    relevance_scores = F1.tolist()
    best_indices = np.argsort(relevance_scores)[::-1][:num_best]  # Obtenez les meilleures reformulations

    best_reformulations = [(generated_sentences[i], relevance_scores[i]) for i in best_indices]
    
    return generated_reformulations, best_reformulations

