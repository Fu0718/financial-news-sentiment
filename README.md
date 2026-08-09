Financial News Sentiment Analysis

A Python-based financial sentiment analysis project using FinBERT to classify financial news into positive, negative, and neutral sentiment.

The project was developed as part of a FinTech / Python portfolio to demonstrate the application of Natural Language Processing (NLP) and machine learning evaluation techniques to financial data.

Project Overview

This project uses ProsusAI/FinBERT, a financial-domain language model, to analyze financial news sentiment.

Instead of testing the model on only a few example sentences, the project evaluates FinBERT on 2,264 financial news articles from the Financial PhraseBank dataset.

The workflow is:

Financial News → FinBERT → Sentiment Prediction → Confidence Score → Model Evaluation → Visualization

Dataset

The project uses the Financial PhraseBank dataset, which contains financial news sentences labeled as:

* Positive
* Negative
* Neutral

The dataset used in this project contains 2,264 articles with high-agreement sentiment labels.

Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Hugging Face Transformers
* FinBERT
* PyTorch
* Apple Silicon MPS acceleration

Model

The sentiment classification model is:

ProsusAI/FinBERT

FinBERT is a BERT-based language model specifically adapted for financial text and sentiment analysis.

For each news article, the model produces:

* Predicted sentiment
* Confidence score

Model Performance

FinBERT was evaluated against the labeled Financial PhraseBank dataset.

Metric Score
Accuracy 97.17%
Macro F1 0.96
Weighted F1 0.97

Classification Performance

Sentiment Precision Recall F1-score
Negative 0.91 0.98 0.94
Neutral 1.00 0.97 0.98
Positive 0.95 0.98 0.96

The model achieved strong performance across all three sentiment categories.

Visualizations

Sentiment Distribution

The project generates a chart showing the distribution of predicted financial news sentiment.

Confidence Distribution

The confidence distribution shows how confident FinBERT was in its predictions.

Confusion Matrix

The confusion matrix compares the model’s predicted sentiment with the original dataset labels.

Project Structure

financial-news-sentiment/
│
├── financial_news_dataset.csv
├── financial_news.csv
│
├── sentiment_analysis.py
│
├── sentiment_results.csv
│
├── sentiment_distribution.png
├── confidence_distribution.png
├── confusion_matrix.png
│
└── README.md

How It Works

1. Load the Financial PhraseBank dataset.
2. Extract the financial news articles.
3. Load the FinBERT sentiment analysis model.
4. Analyze each financial news article.
5. Generate sentiment predictions and confidence scores.
6. Compare predictions with the original dataset labels.
7. Calculate accuracy, precision, recall, and F1-score.
8. Generate visualization charts.
9. Save the prediction results to a CSV file.

How to Run

Clone the repository and navigate to the project directory.

Install the required Python packages:

pip install pandas matplotlib scikit-learn transformers torch

Run the analysis:

python sentiment_analysis.py

The program will generate:

sentiment_results.csv
sentiment_distribution.png
confidence_distribution.png
confusion_matrix.png

Key Findings

The model achieved an overall accuracy of 97.17% on the Financial PhraseBank dataset.

The results demonstrate that a finance-specific NLP model such as FinBERT can effectively classify financial news sentiment.

The confusion matrix and classification report also provide a more detailed view of model performance across positive, negative, and neutral financial news.

Future Improvements

Potential extensions of this project include:

* Applying sentiment analysis to real-time financial news
* Connecting the model to a financial news API
* Analyzing sentiment changes over time
* Combining sentiment signals with stock price data
* Building a simple financial sentiment dashboard
* Exploring whether news sentiment can be used as a trading signal

Purpose

This project was created as part of my FinTech / Python portfolio to demonstrate practical experience with:

* Financial data
* Natural Language Processing
* Machine learning models
* Model evaluation
* Data visualization
* Python programming