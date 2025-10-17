# Fine-tuning Pretrained Transformers on Social Media Sentiment Analysis
Jiang Yifan DSA4213 Assignment3

# Fine-tuning Pretrained Transformers on Social Media Sentiment Analysis

## Project Overview
This project focuses on fine-tuning pretrained transformer models for sentiment analysis on social media data. The dataset used is **Sentiment140**, which contains labeled tweets for positive and negative sentiment classification.

## Dataset
The dataset can be downloaded from Kaggle: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

After downloading, place the dataset file in the project directory or adjust the path in `main.py` accordingly.

## Installation

Make sure you have **Python** installed.  
Install the required packages using pip:

```bash
pip install transformers==4.44.2
pip install datasets==2.19.1
pip install evaluate==0.4.2
pip install peft==0.11.1
pip install accelerate==0.33.0
pip install "huggingface-hub>=0.23,<0.25"
pip install "tokenizers>=0.19,<0.21"
pip install safetensors>=0.4
```
Or, if you prefer, run the following in Python:
```Python
packages = [
    "transformers==4.44.2",
    "datasets==2.19.1",
    "evaluate==0.4.2",
    "peft==0.11.1",
    "accelerate==0.33.0",
    "huggingface-hub>=0.23,<0.25",
    "tokenizers>=0.19,<0.21",
    "safetensors>=0.4"
]

for pkg in packages:
    !pip install -q --no-warn-conflicts "{pkg}"
```

## Usage

Run the main program with:

```bash
python main.py
```

Make sure to adjust any dataset paths or hyperparameters directly in main.py as needed.
