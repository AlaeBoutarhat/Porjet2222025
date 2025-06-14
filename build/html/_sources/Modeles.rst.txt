Modeles
++++++++
.. footer::
   :class: rst-footer-buttons

   :doc:`Previous <introduction>` | :doc:`Next <Building>`

Keyword Extraction Model:
--------------------------------
For the keyword extraction task, we use the Zephyr 7B Beta model provided by Hugging Face. 
This is a powerful instruction-tuned large language model designed for high-performance NLP tasks.

The model was fine-tuned on a custom dataset specifically designed for the keyword extraction 
use case. Each training example consists of:

**An input prompt:** representing a user query; Preceded by a directive prompt to guide the model during fine-tuning,

**An output:** consisting of the list of objects (or keywords) mentioned in the prompt.

This fine-tuning approach allows the model to learn how to identify 
and extract relevant objects regardless of the prompt’s structure, wording, 
or semantic style. After training, the Zephyr model is capable of handling diverse 
user instructions and reliably returning the corresponding keywords, making it a robust component 
for real-world prompt interpretation in image segmentation tasks.
