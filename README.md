# Topic Modeling Pipeline for Amazon Fine Foods Reviews

This project implements a comprehensive NLP topic modeling pipeline using the Amazon Fine Foods Reviews dataset.

## Features

- **Text Preprocessing**: Tokenization, stopword removal, bigram/trigram phrase extraction
- **Vectorization**: TF-IDF vectorization for topic modeling
- **Topic Modeling**: LDA and NMF algorithms
- **Evaluation**: Topic coherence and perplexity metrics
- **Vector Database**: FAISS integration for retrieving representative reviews
- **Visualization**: Interactive topic visualizations and word clouds

## Project Structure

```
├── data/                    # Dataset storage
├── notebooks/               # Jupyter notebooks for analysis
├── src/                    # Source code
│   ├── preprocessing.py    # Text preprocessing pipeline
│   ├── modeling.py        # Topic modeling algorithms
│   ├── evaluation.py      # Model evaluation metrics
│   ├── visualization.py   # Plotting and visualization
│   └── utils.py           # Utility functions
├── results/               # Output results and visualizations
└── requirements.txt       # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Dataset Setup

The Amazon Fine Foods Reviews dataset is not included in this repository due to size limitations. You need to download it separately:

1. Download the dataset from [Amazon Fine Foods Reviews](https://snap.stanford.edu/data/web-FineFoods.html)
2. Place the `finefoods.txt.gz` file in the `data/` directory
3. Extract the file: `gunzip data/finefoods.txt.gz`

The dataset contains customer reviews of fine foods from Amazon and is approximately 353MB in size.

## Usage

Run the main pipeline:
```bash
python src/main.py
```

Or explore interactively:
```bash
jupyter notebook notebooks/topic_modeling_analysis.ipynb
```

## Dataset

The Amazon Fine Foods Reviews dataset contains customer reviews of fine foods from Amazon. The dataset includes:
- Review text
- Product ratings
- Helpfulness scores
- Timestamps

## Output

The pipeline generates:
- Top words per topic
- Topic coherence and perplexity scores
- Interactive topic visualizations
- Representative reviews for each topic
- Model comparison results
