from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

# Load FinBERT sentiment analysis model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

print("FinBERT model loaded successfully!")
# Sample financial news
news = [
    "Apple reported stronger-than-expected quarterly earnings.",
    "NVIDIA shares fell sharply after disappointing revenue guidance.",
    "Microsoft announced a new partnership with a major technology company.",
    "Amazon's stock remained unchanged after the earnings report."
]

# Analyze sentiment
results = sentiment_pipeline(news)

for text, result in zip(news, results):
    print("\nNews:", text)
    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"], 4))

    # Create a DataFrame
df = pd.DataFrame({
    "News": news,
    "Sentiment": [result["label"] for result in results],
    "Confidence": [result["score"] for result in results]
})

print("\nSentiment Analysis Results:")
print(df)

# Count sentiment categories
sentiment_counts = df["Sentiment"].value_counts()

print("\nSentiment Counts:")
print(sentiment_counts)

# Plot sentiment distribution
plt.figure(figsize=(8, 5))

sentiment_counts.plot(kind="bar")

plt.xlabel("Sentiment")
plt.ylabel("Number of News Articles")
plt.title("Financial News Sentiment Distribution")

plt.tight_layout()
plt.savefig("sentiment_distribution.png", dpi=300)

plt.show()