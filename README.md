Question Answering Systems in NLP
 This repository contains a Jupyter notebook that explores various types of Question Answering (QA)
 systems using Natural Language Processing (NLP) techniques. The project is a part of the
 Self-Directed Learning (SDL) module and demonstrates both theoretical understanding and practical
 implementation.
 File Structure- QA_systems_NLP_SD.ipynb - The main notebook showcasing:
  - Closed-domain and open-domain QA
  - Extractive QA using HuggingFace Transformers
  - Reader-Ranker pipeline
  - Pre-trained models and pipelines
  - Sample evaluations and outputs
 Getting Started
 Prerequisites:
 pip install transformers
 pip install torch
 pip install datasets
 pip install jupyter
 Running the Notebook:
1. Clone this repository:
   git clone https://github.com/yourusername/qa-systems-nlp.git
   cd qa-systems-nlp
 2. Launch the Jupyter Notebook:
   jupyter notebook QA_systems_NLP_SD.ipynb
 Key Concepts Covered- Extractive Question Answering- Closed vs Open-domain QA- HuggingFace Pipelines: Using pipeline("question-answering")- Pre-trained Models: distilbert-base-cased-distilled-squad,
 bert-large-uncased-whole-word-masking-finetuned-squad, etc.- Answer Extraction from contexts using modern transformers
 References- Hugging Face Documentation: https://huggingface.co/docs- SQuAD Dataset: https://rajpurkar.github.io/SQuAD-explorer/
