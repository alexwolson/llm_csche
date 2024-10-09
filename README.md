# Link: https://github.com/alexwolson/llm_csche
# Email: alex.olson@utoronto.ca

# SMILES to Chemical Formula Translator - Fine-Tuning Tutorial

This repository contains a Jupyter notebook that demonstrates how to fine-tune a pre-trained language model to translate SMILES (Simplified Molecular Input Line Entry System) strings into chemical formulas. This tutorial was designed as part of a workshop, introducing participants to the concept of transfer learning in the context of chemical informatics.

**Quick Start on Google Colab**

You can run the notebook directly in Google Colab by clicking on the following link:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexwolson/llm_csche/blob/main/SMILES_to_Chemical_Formula_Translator_Fine_Tuning_Tutorial.ipynb)

## Overview

The primary objective of this tutorial is to take an existing language model and train it to perform a specialized task: converting SMILES strings, a text-based representation of molecular structures, into chemical formulas. This is accomplished using tools from the HuggingFace ecosystem, which provides the `transformers` library for working with state-of-the-art NLP models.

The tutorial involves the following steps:
- Setting up the Python environment and installing necessary packages.
- Loading pre-trained transformer models.
- Fine-tuning the model on a custom dataset of SMILES strings and corresponding chemical formulas.
- Evaluating the performance of the trained model.

### Key Tools and Libraries Used:
- **[Transformers](https://github.com/huggingface/transformers)**: The main library for working with transformer models.
- **[Datasets](https://github.com/huggingface/datasets)**: A library for loading and processing datasets.
- **PyTorch**: The underlying deep learning framework used by the `transformers` library.
- **Pandas**: For handling and displaying tabular data.
- **Levenshtein**: For calculating the similarity between the predicted chemical formulas and the ground truth.

## Requirements

To run the notebook, you will need:

- Python 3.7 or later.
- The following libraries:
    - `transformers`
    - `datasets`
    - `accelerate`
    - `torch`
    - `pandas`
    - `python-Levenshtein`

You can install these packages using:

```sh
pip install -U transformers datasets accelerate torch pandas python-Levenshtein
```

## Running the Notebook

To run the notebook, you can either clone this repository and open the notebook in Jupyter, or you can load the notebook in Google Colab.

### Running Locally

1. Clone this repository to your local machine:
    
    ```sh
    git clone https://github.com/alexwaolson/llm_csche.git
   ```
   
2. Navigate to the cloned directory:

    ```sh
    cd llm_csche
    ```
   
3. Install the required packages:

    ```sh
    pip install -U transformers datasets accelerate torch pandas python-Levenshtein
    ```
    
    Alternately, using Conda:
    
    ```sh
   conda install -c huggingface transformers datasets accelerate torch pandas python-Levenshtein 
    ```
   
   
4. Launch Jupyter Notebook:

    ```sh
    jupyter notebook
    ```
   
### Running on Google Colab

You can open the notebook directly in Google Colab by clicking on the following link:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexwolson/llm_csche/blob/main/SMILES_to_Chemical_Formula_Translator_Fine_Tuning_Tutorial.ipynb)

## Workshop Objectives

The interactive session of this tutorial is intended to introduce participants to:

- Fine-tuning pre-trained transformer models for domain-specific tasks.
- Working with SMILES strings and chemical data.
- Evaluating model predictions using metrics like Levenshtein distance.

By the end of this workshop, participants should have a foundational understanding of how to take a general-purpose language model and adapt it to a specialized application within the field of chemistry.

## Acknowledgements

- HuggingFace for their amazing transformers and datasets libraries.
- Participants of the CSChE 2024 Conference for their insightful questions and participation.

## Contributing

Feel free to fork the repository and submit pull requests. Contributions, whether through bug reports, new features, or documentation improvements, are highly welcomed!
