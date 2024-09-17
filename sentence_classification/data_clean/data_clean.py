import pandas as pd
import matplotlib.pyplot as plt

# Try reading the CSV file with a different encoding (ISO-8859-1)
df = pd.read_csv('/home/vision/workspace/sonnt373/dev/NLP_Research/sentence_classification/dataset/sentiment-analysis-dataset/test.csv', encoding='ISO-8859-1')

# Group by 'sentiment' and get counts
sentiment_counts = df['sentiment'].value_counts()

# Plot the distribution of sentiment labels
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color='skyblue')
plt.title('Sentiment Label Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('sentiment_plot.png')
plt.show()
