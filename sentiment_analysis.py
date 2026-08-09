from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load FinBERT sentiment analysis model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

print("FinBERT model loaded successfully!")

# Load Financial PhraseBank dataset
df = pd.read_csv(
    "financial_news_dataset.csv"
)

print("\nDataset loaded successfully!")
print("Number of news articles:", len(df))

# Get financial news
news = df["News"].tolist()

print("\nRunning FinBERT sentiment analysis...")

# Analyze all news articles
results = sentiment_pipeline(
    news,
    batch_size=16,
    truncation=True
)

# Add FinBERT predictions to DataFrame
df["Predicted_Sentiment"] = [
    result["label"] for result in results
]

df["Confidence"] = [
    result["score"] for result in results
]

print("\nSentiment Analysis Results:")
print(df.head())

# Calculate model accuracy
accuracy = accuracy_score(
    df["True_Sentiment"],
    df["Predicted_Sentiment"]
)

print("\nModel Accuracy:")
print(round(accuracy, 4))

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        df["True_Sentiment"],
        df["Predicted_Sentiment"]
    )
)

# Save prediction results
df.to_csv(
    "sentiment_results.csv",
    index=False
)

print("\nPrediction results saved to sentiment_results.csv")

# Create confusion matrix
cm = confusion_matrix(
    df["True_Sentiment"],
    df["Predicted_Sentiment"],
    labels=["positive", "neutral", "negative"]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Positive", "Neutral", "Negative"]
)

disp.plot()

plt.title("FinBERT Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

# Sentiment distribution
sentiment_counts = df["Predicted_Sentiment"].value_counts()

plt.figure(figsize=(8, 5))

sentiment_counts.plot(kind="bar")

plt.xlabel("Predicted Sentiment")
plt.ylabel("Number of News Articles")
plt.title("Financial News Sentiment Distribution")

plt.tight_layout()

plt.savefig(
    "sentiment_distribution.png",
    dpi=300
)

plt.show()

# Confidence distribution
plt.figure(figsize=(8, 5))

df["Confidence"].plot(
    kind="hist",
    bins=20
)

plt.xlabel("Confidence Score")
plt.ylabel("Number of News Articles")
plt.title("FinBERT Confidence Score Distribution")

plt.tight_layout()

plt.savefig(
    "confidence_distribution.png",
    dpi=300
)

plt.show()

print("\nAnalysis completed successfully!")