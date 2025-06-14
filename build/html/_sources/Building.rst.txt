Builiding
++++++++++
.. footer::
   :class: rst-footer-buttons

   :doc:`Previous <introduction>` | :doc:`Next <Building>`


**Part 1 : Keyword Extraction Model**
#########################################
In this section, we will explore how to create the dataset for fine-tuning the keyword 
extraction model, as well as the testing and validation of this model


Data Creation
=================

**Dataset**
----------------------

We will be working with the Object365 dataset, available on Hugging Face.

The Objects365 dataset is a large-scale, high-quality dataset specifically created to advance research in object detection, with a strong emphasis on the diversity of objects as they appear in real-world scenarios ("in the wild").
- Scale: It comprises approximately 2 million high-resolution images.

- Object Categories: The dataset covers 365 distinct object categories. These categories are chosen to be common and varied, representing a wide range of everyday objects.

- Annotations: Each image is richly annotated with over 30 million high-quality bounding boxes. This means that numerous object instances are labeled within each image.

- Annotation Details: The annotations include precise bounding boxes outlining each object and a corresponding category label for each box.

- Purpose: Objects365 serves as a challenging and comprehensive benchmark for training and evaluating object detection models. Its scale and diversity aim to push the boundaries of current object detection research.

- High-Quality Annotations: The annotations are meticulously created through a carefully designed three-step annotation pipeline to ensure accuracy.

- Focus on Real-World Objects: Unlike some datasets that might focus on more iconic or centered objects, Objects365 includes objects in various contexts, poses, and occlusions, reflecting real-world complexity.

- Benefits for Pre-training: Models pre-trained on Objects365 have demonstrated superior performance and better generalization on various downstream object detection tasks compared to models pre-trained on datasets like ImageNet.

- Dataset Structure: The dataset typically consists of a collection of images and a corresponding annotation file (often in a format similar to COCO), detailing the bounding box coordinates and category labels for each object in each image.

- Usage: It is widely used for training advanced deep learning models for object detection, evaluating their performance, and researching challenging aspects like detecting rare objects and improving model generalization.

**Data to fine tune the model**
--------------------------------------------

To fine-tune the Zephyr 7B Beta model, we first build a custom training dataset derived from the Object365 dataset. The process involves several key steps:

:blue-bold:`Annotation Extraction:` We begin by extracting the object annotations available in Object365. Each annotation corresponds to one or more object classes present in an image.

:blue-bold:`Prompt Generation:` We generate around 100 diverse prompts that vary in form, semantics, grammatical structure, and complexity. These prompts are designed to reflect the variety of ways a user might request object segmentation.

:blue-bold:`Prompt Annotation Pairing:` Each prompt contains a placeholder where the list of annotations (object classes) will be inserted. We randomly associate each prompt with a different list of annotations, simulating realistic user instructions.

:blue-bold:`Structuring the Dataset:` The resulting data is formatted as a structured JSON file, where:

- The input field contains the generated prompt with embedded annotations, preceded by a directive prompt that guides the model during fine-tuning.

- The output field contains the corresponding list of object classes extracted from the prompt.

This approach allows the model to learn how to identify keywords from a wide variety of user expressions, enhancing its generalization capabilities for real-world use cases.

{"input": "\nInstruction: You are an assistant specialized in object segmentation. Your task is to list all the objects mentioned in the description, following the specific instructions provided.\nExample: For the input 'Segment cats, dogs, but not humans', the expected output is ['cat', 'dog'].\nDescription: Correctly annotate elements of type Vase Flower Picture Frame Cabinet shelf Coffee Table Couch Pillow Trash bin Can Moniter TV Remote\n", "output": ["Cabinet/shelf", "Coffee Table", "Couch", "Flower", "Moniter/TV", "Picture/Frame", "Pillow", "Remote", "Trash bin Can", "Vase"]}
{"input": "\nInstruction: You are an assistant specialized in object segmentation. Your task is to list all the objects mentioned in the description, following the specific instructions provided.\nExample: For the input 'Segment cats, dogs, but not humans', the expected output is ['cat', 'dog'].\nDescription: Segment each Person Microphone found in this scene\n", "output": ["Microphone", "Person"]}


*Iport the dataset*

.. code-block:: python

    from datasets import load_dataset
    ds = load_dataset("jxu124/objects365")
    print(ds)

*Split the dataset into train and test data*

.. code-block:: python

    train_data = ds['train']
    print(train_data[0])
    test_data = ds['validation']
    print(test_data[0])

*Creat prompts and randomly associate a prompt with a list of annotations*

.. code-block:: python
   

   import random
   import json

   # Sélectionner un échantillon aléatoire de 400 000 indices
   sample_indices = random.sample(range(len(train_data)), 200000)

   # Créer un sous-ensemble à partir de ces indices
   sample_data = [train_data[i] for i in sample_indices]

   # Liste pour stocker les prompts
   prompts = []

   # Liste des modèles de prompt
   prompt_templates = [
       # 🔹 Simple prompts
       "Detect and segment the following objects in the image: {}.",
       "Identify and locate the following elements: {}.",
       "What objects are visible in the image? Answer: {}.",
       "Precisely segment the objects: {}.",
       "Recognize and annotate the following elements: {}.",
       "List all objects present in the image, including {}.",
       "Find and mark the visible elements: {}.",
       "Accurately locate the objects: {}.",
       "Separate and distinguish the following objects: {}.",
       "Draw the contours of each {} in the image.",

       # 🔹 Question-based prompts
       "What detectable objects are in the image? {}.",
       "Can {} be seen in this image?",
       "Which elements in the image belong to the category {}?",
       "Does the image contain {}?",
       "What distinct objects are present, including {}?",
       "Describe all visible objects, particularly {}.",
       "How many {} are present in the image?",
       "Does the image depict a scene containing {}?",
       "Which elements are the most visible, including {}?",
       "What is the main object in the image among {}?",

       # 🔹 Negation-based prompts
       "Do not consider objects other than {}.",
       "Ignore elements that are not {}.",
       "The image does NOT contain {}. Identify only the other objects.",
       "Include only {} in the analysis.",
       "Avoid detecting anything except {}.",
       "Do not segment any objects other than {}.",
       "Exclude elements that do not belong to category {}.",
       "Detect all objects except {}.",
       "Do not consider objects that are not {}.",
       "Filter only for the presence of {}.",

       # 🔹 Action-specific prompts
       "Draw a box around {}.",
       "Highlight the area containing {}.",
       "Outline the exact shape of {}.",
       "Emphasize the presence of {} in the image.",
       "Create a segmentation mask for {}.",
       "Precisely define {}.",
       "Classify the objects including {}.",
       "Correctly annotate elements of type {}.",
       "Add a label for each {}.",
       "Group objects similar to {}.",

       # 🔹 Detailed description prompts
       "Describe in detail the following objects in the image: {}.",
       "Provide an explanation of the presence of {}.",
       "Analyze the image and precisely identify {}.",
       "Classify the detected objects, including {}.",
       "Associate each {} with its exact position in the image.",
       "What types of objects appear here? Included list: {}.",
       "Detail the shape and color of {}.",
       "Which objects are closest to {}?",
       "Explain how {} interacts with other objects.",
       "Summarize the objects present, focusing on {}.",

       # 🔹 Contextual or temporal prompts
       "How far is {} from other objects?",
       "Is there any overlap between {} and other elements?",
       "How is {} positioned in the image?",
       "Determine if {} is in the foreground or background.",
       "Observe interactions between {} and other objects.",
       "Analyze the proximity between {} and its environment.",
       "Check if {} is in motion or static.",
       "See if {} is partially hidden by other objects.",
       "Detect if {} is reflected on a surface.",
       "Compare the size of {} with other objects present.",

       # 🔹 Alternative phrasing prompts
       "Indicate the position of {} in the image.",
       "Segment each {} found in this scene.",
       "Which categories of objects are represented, including {}?",
       "Does the image contain more {} than other objects?",
       "Classify the detected objects, focusing on {}.",
       "Identify all {} and estimate their relative size.",
       "Which objects are smaller or larger than {}?",
       "Count the exact number of {}.",
       "Define the outline of {} in this image.",
       "Locate {} and mark their precise placement.",

       # 🔹 Uncertainty-based prompts
       "Is it possible that the image contains {}?",
       "Detect all visible objects, assuming the presence of {}.",
       "List the potential objects in the image, including {}.",
       "Does the object {} appear well-defined in the image?",
       "Find identifiable objects, focusing on {}.",
       "Estimate the presence of {} among the visible elements.",
       "What is the confidence level that {} is in the image?",
       "Detect the objects with the highest probability, including {}.",
       "Which objects could be confused with {}?",
       "Is {} fully visible or partially hidden?"
   ]

   # Générer les prompts pour l'échantillon
   for data in sample_data:
       image_id = data['global_image_id']
       image_path = data['image_path']
       anns_info = data['anns_info']

       # Initialiser la liste des objets pour cette image
       objects = []

       # Pour chaque annotation (objet) de l'image
       for ann in anns_info:
           category = ann['category']
           objects.append(category)

       # Générer un prompt brut avec les objets
       prompt_brut = ", ".join(objects)

       # Choisir un prompt template au hasard
       prompt_template = random.choice(prompt_templates)
       prompt = prompt_template.format(prompt_brut)

       # Correction du prompt
       prompt_corrige = prompt.replace("the following objects", "the following items")

       # Mots-clés extraits de l'objet
       keywords = objects

       # Ajouter le prompt à la liste
       prompts.append({
           "prompt_brut": prompt,
           "prompt_corrige": prompt_corrige,
           "keywords": keywords,
           "image_id": image_id,
           "image_path": image_path
       })

   # Sauvegarder les prompts par lots (par exemple, 1000 prompts par fichier)
   batch_size = 1000
   batch_num = 0

   # Sauvegarder chaque lot dans un fichier JSON
   for i in range(0, len(prompts), batch_size):
       batch_prompts = prompts[i:i+batch_size]
       batch_file = f'/content/drive/MyDrive/ProjetMetier/Data1/prompts_batch_{batch_num}.json'

       # Sauvegarder le lot dans un fichier JSON
       with open(batch_file, 'w', encoding='utf-8') as f:
           json.dump(batch_prompts, f, ensure_ascii=False, indent=4)

       print(f"Lot {batch_num} sauvegardé : {len(batch_prompts)} prompts")
       batch_num += 1



**Fine tune the modle on the created dataset**
--------------------------------------------------

Fine-tuning a large language model such as Zephyr or Mistral requires significant computational resources. Typically, a GPU with at least 24 GB of VRAM is recommended for efficiently training a 7B parameter model. However, using optimization techniques such as LoRA and 4-bit quantization (bnb_4bit), it is possible to fine-tune on more modest hardware—sometimes with as little as 12–16 GB of VRAM, or even 8 GB with careful memory management. Additionally, at least 16 to 32 GB of RAM is advised depending on the size of the dataset and the model architecture.

Several cloud platforms support fine-tuning:

- RunPod.io: Affordable GPU rental service (A100, V100, T4, etc.) with support for Jupyter Notebooks.

- Google Colab Pro/Pro+: An accessible option for smaller-scale training, offering GPUs like T4 or A100 (based on availability).

- Paperspace: Provides notebooks with powerful GPUs on demand.

- Lambda Labs, NVIDIA LaunchPad, and Hugging Face Training Cluster are also suitable for more extensive training tasks.

For local fine-tuning, setups with GPUs between 8–12 GB VRAM can still be viable using lightweight fine-tuning frameworks such as PEFT (LoRA), Transformers + Accelerate, and memory-efficient tools like DeepSpeed, QLoRA, or bitsandbytes. However, training locally on limited 
hardware requires careful configuration to avoid out-of-memory errors

After the fine-tuning is complete, we will have a folder containing the files obtained after the fine-tuning. 
This folder and the original template are used to extract keywords from the prompts.


**Testing the modele**
-------------------------

.. code-block:: python

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    def segment_objects_with_prompting(prompt,
                                        base_model_name="HuggingFaceH4/zephyr-7b-beta",
                                        lora_path="/teamspace/studios/this_studio/phi2/zypher",
                                        max_new_tokens=128):
        # Instructions système
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

        # Prompt complet
        full_prompt = f"<|system|>\n{system_instruction}\n<|user|>\n{prompt}\n<|assistant|>\n"

        # Forçage du CPU
        device = torch.device("cpu")

        # Chargement du modèle de base et du tokenizer sur CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map={"": device}
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Chargement du modèle LoRA sur CPU
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.to(device)
        model.eval()

        # Encodage
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

        # Génération
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_return_sequences=1,
            )

        # Décodage et extraction
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = result.split("<|assistant|>")[-1].strip()

        # Nettoyage des objets
        lines = [line.strip() for line in assistant_response.splitlines() if line.strip()]
        if lines:
            object_line = lines[0]
            object_candidates = [obj.strip().capitalize() for obj in object_line.split(',') if obj.strip()]
            cleaned_objects = list(set(object_candidates))
        else:
            cleaned_objects = []

        print("🎯 Objets détectés :", cleaned_objects)
        return cleaned_objects

    # Exemple d’appel
    segment_objects_with_prompting("Segment cats, dogs, and birds but ignore cars and chairs.")
    segment_objects_with_prompting("Segment all animals visible in the image like horses and elephants but ignore the background")
    segment_objects_with_prompting("Segment everything related to food but ignore drinks and containers.")



**Part 2 : Preprocessing of the prompt**
#########################################


**I. First approach : NLP functions**
======================================

Our focus here is on the intelligent prompt preprocessing system, a critical component of the Segma Vision Pro Light pipeline. We will detail its methodology for transforming raw user input into clear,
actionable instructions, thereby enhancing system efficiency and accuracy

The main objectives are:

- 1. Cleaning and Normalization: Remove noise (special characters, case inconsistencies) from prompts.

- 2. Grammar Correction: Fix grammatical errors in French or English text.

- 3. Translation: Translate text between French and English.

- 4. Paraphrasing: Generate diverse and semantically equivalent reformulations of a prompt, ranked by relevance.

Detailed Breakdown of Each Function

**1. nettoyer_prompt(prompt):**
------------------------------------------
This function leans a text prompt by removing special characters and converting it to lowercase

Purpose
~~~~~~~
Cleans a prompt by removing special characters and converting to lowercase.

Techniques
~~~~~~~~~~
- Regex pattern ``[^\w\s]`` to remove non-alphanumeric chars
- String method ``lower()`` for case normalization

Input/Output
~~~~~~~~~~~~
+----------+---------------------------------+
| Input    | Raw prompt (string)             |
+----------+---------------------------------+
| Output   | Cleaned lowercase text (string) |
+----------+---------------------------------+

Example::
  
  input: "Hello! How's the weather?"
  output: "hello hows the weather"

**2. corriger_phrase():**
----------------------------

Purpose
~~~~~~~
Corrects grammatical errors in French or English text.

Techniques
~~~~~~~~~~
- ``language_tool_python`` library
- Language-specific rules ('fr' or 'en-US')

Parameters
~~~~~~~~~~~~~~~
+----------+---------------------+-----------+
| Name     | Type                | Default   |
+----------+---------------------+-----------+
| phrase   | str                 | Required  |
+----------+---------------------+-----------+
| langue   | str ('fr'/'en')     | 'fr'      |
+----------+---------------------+-----------+

Example::
  
  input: "Je suis aller"
  output: "Je suis allé"

**3. Translation Functions:**
--------------------------------

3.1 traduire_en_anglais()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Purpose
~~~~~~~~~~~
Translates French text to English.

3.2 traduire_en_francais()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Purpose
~~~~~~~
Translates English text to French.

Techniques
~~~~~~~~~~
- ``GoogleTranslator`` from deep_translator
- Automatic language detection (source='auto')

Example::
  
  input: "Bonjour"
  output: "Hello"

**4. reform():**
-------------------

Purpose
~~~~~~~~~~~
Generates and ranks paraphrased versions of a prompt.

Technical Implementation
~~~~~~~~~~~~~~~~~~~~~~~~
1. Paraphrase Generation:
   - T5 model (Vamsi/T5_Paraphrase_Paws)
   - Advanced sampling: temperature=1.8 , top_p=0.8 , repetition_penalty=1.2

2. Semantic Ranking:
   - BERTScore evaluation
   - roberta-large model
   - F1 metric scoring

Parameters
~~~~~~~~~~
+-------------------+---------------------+-----------+
| Name              | Type                | Default   |
+-------------------+---------------------+-----------+
| prompt            | str                 | Required  |
+-------------------+---------------------+-----------+
| dataset_path      | str                 | Required  |
+-------------------+---------------------+-----------+
| num_reformulations| int                 | 10        |
+-------------------+---------------------+-----------+
| num_best          | int                 | 6         |
+-------------------+---------------------+-----------+

Output
~~~~~~
Tuple containing:
1. All generated reformulations (list)
2. Top-ranked reformulations with scores (list of tuples)

Dependencies
------------
- regex, language_tool_python, deep_translator
- transformers, torch, bert_score, numpy


.. note::
   Source file : :download:`Preprocessing_functions.py`


**II. Second approach : LLM-Based Text Processing Approach**
==================================================================

This approach replaces multiple NLP functions with a single Large Language Model (LLM) for unified text preprocessing, offering enhanced quality and simplicity.

**Key Advantages**
---------------------

1. End-to-End Processing

- Replaces multiple specialized functions with one model
- Eliminates complex pipelines and tool dependencies

2. Contextual Understanding

- Better grammar correction through semantic analysis
- Maintains coherent paraphrasing with original intent

3. Multilingual Efficiency

- Supports 100+ languages natively
- Provides more natural translations than dictionary methods

4. Adaptability

- Customizable via prompt engineering
- Supports few-shot learning for domain adaptation

**LLM Model Selection**
--------------------------------------------------


The **Mistral** model, run locally via the **Ollama** environment, was chosen as the primary engine for processing user prompts. Several technical and operational criteria informed this decision:

- Advanced Capabilities: Mistral is recognized for its advanced text understanding and generation capabilities, particularly in tasks like reformulation, grammatical correction, and translation. This makes it ideally suited for our intelligent semantic preprocessing needs.

- Open-Source Nature: As an open-source model, Mistral offers flexibility and transparency.

- Seamless Integration: The model integrates easily with evaluation tools such as BERTScore, enabling automated measurement of the relevance of generated reformulations.


**Utilizing BERTScore for Reformulation Filtering**
---------------------------------------------------------

What is BERTScore?

BERTScore is a metric derived from the F1 score of BERT, specifically utilizing a roberta-large model. It's designed to measure the semantic 
similarity between original prompts and their generated reformulations.


Following the generation of initial prompt reformulations by the LLM model (Mistral via Ollama), it's essential to filter and select those most relevant and semantically closest to our dataset used for training the keyword extraction model (the classes to be segmented or labeled). To achieve this, we've integrated an advanced scoring method: **BERTScore**