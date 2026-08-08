Financial News Sentiment Analysis

A Python project that uses FinBERT to analyze the sentiment of financial news articles.

Project Overview

This project applies a finance-specific NLP model, FinBERT, to classify financial news into three sentiment categories:

* Positive
* Negative
* Neutral

The model also provides a confidence score for each prediction.

Technologies Used

* Python
* Pandas
* Matplotlib
* Hugging Face Transformers
* FinBERT
* Apple Silicon MPS acceleration

How It Works

The project follows this workflow:

Financial News → FinBERT → Sentiment Classification → Confidence Score → DataFrame → Visualization

Each news article is analyzed by FinBERT and assigned a sentiment category together with its prediction confidence.

Example Results

The project currently analyzes four financial news examples:

News Sentiment Confidence
Apple reported stronger-than-expected quarterly earnings. Positive 0.9372
NVIDIA shares fell sharply after disappointing revenue guidance. Negative 0.9740
Microsoft announced a new partnership with a major technology company. Positive 0.8794
Amazon’s stock remained unchanged after the earnings report. Neutral 0.5642

Sentiment Distribution

The current dataset contains:

* Positive: 2
* Negative: 1
* Neutral: 1

Project Structure

financial-news-sentiment/
│
├── sentiment_analysis.py
├── sentiment_distribution.png
└── README.md

How to Run

Clone the repository and run:

python3 sentiment_analysis.py

The script performs sentiment analysis and generates the sentiment distribution chart.

Purpose

This project was created as part of my FinTech / Python portfolio to demonstrate the application of Natural Language Processing (NLP) to financial data.