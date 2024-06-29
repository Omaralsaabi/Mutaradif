# Synonym Generator Tool

The Synonym Generator is a Python tool designed to generate and refine synonyms for Arabic words using the Ollama Engine. It allows users to specify a word, model, number of synonyms, and a similarity threshold to retrieve the most relevant synonyms based on cosine similarity.

## Features

- Generate synonyms for Arabic words.
- Refine synonyms based on cosine similarity.
- Use the Ollama engine model for accurate and relevant results.

## Installation

Follow these steps to set up and use the Synonym Generator tool on your system:

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Setup Environment

Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Ollama Engine Setup

Pull the desired model from the Ollama library. Visit the [Ollama Library](https://ollama.com/library) to see available models and further details:

```bash
ollama pull aya  # Replace 'aya' with any other model you might want to use
```

## Usage

To use the Synonym Generator tool, you can run the script with the desired parameters:

```bash
python3 run.py --word "your arabic word" --model "aya" --num_synonyms 10 --similarity_threshold 0.9
```

Replace `"your arabic word"` with the Arabic word you want synonyms for.

### Command-Line Arguments

- `--word`: The Arabic word you want to generate synonyms for. (required)
- `--model`: The Ollama model used for generating synonyms. Default is `"aya"`.
- `--num_synonyms`: The number of synonyms to generate. Default is `10`.
- `--similarity_threshold`: The cosine similarity threshold for selecting the best synonyms. Default is `0.9`.

## Output

The tool will print a list of synonyms that meet the specified similarity threshold.