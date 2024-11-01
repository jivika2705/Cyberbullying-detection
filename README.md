import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import re


df = pd.read_csv('/content/cyberbullying_tweets.csv')
df.head()

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define function to remove stopwords
def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

# Apply stopword removal and create a new column
data['cleaned_tweet'] = data['tweet_text'].apply(remove_stopwords)

df = df.dropna(subset=['cyberbullying_type'])
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_tweet'], df['cyberbullying_type'], test_size=0.2, random_state=42)

# Convert text to numerical data
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression())

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_log_reg = model.predict(X_test)

unique_labels = df['cyberbullying_type'].unique() # Assuming your target variable column is named 'cyberbullying_type'

# Print the classification report with the correct target names
print(classification_report(y_test, y_pred_log_reg, target_names=unique_labels))

# Create and train the SVM model
from sklearn.svm import SVC
model = SVC(kernel='linear') 
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred_svm = model.predict(X_test)  # Use X_test instead of X_test_tfidf

unique_labels = df['cyberbullying_type'].unique()  # Assuming your target variable column is named 'cyberbullying_type'

# Print the classification report with the correct target names
print(classification_report(y_test, y_pred_svm, target_names=unique_labels))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
unique_labels = df['cyberbullying_type'].unique()  # Assuming your target variable column is named 'cyberbullying_type'

# Print the classification report with the correct target names
print(classification_report(y_test, y_pred, target_names=unique_labels))

metrics_log_reg = evaluate_model(y_test, y_pred_log_reg)
metrics_random_forest = evaluate_model(y_test, y_pred_rf)
metrics_svm = evaluate_model(y_test, y_pred_svm)


results = {
    'Logistic Regression': metrics_log_reg,
    'Random Forest': metrics_random_forest,
    'Support Vector Machine': metrics_svm,
}
print("Evaluation Results (Accuracy, Precision, Recall, F1-score):")
for model_name, scores in results.items():
    print(f"\n{model_name}:")
    # Accessing elements of the scores tuple using indexing
    print(f"  Accuracy: {scores[0]:.4f}")
    print(f"  Precision: {scores[1]:.4f}")
    print(f"  Recall: {scores[2]:.4f}")
    print(f"  F1-score: {scores[3]:.4f}")


import matplotlib.pyplot as plt
import numpy as np

metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Logistic Regression': metrics_log_reg,
    'Random Forest': metrics_rf,
    'SVM': metrics_svm
}

# Convert the dictionary to a DataFrame
df_metrics = pd.DataFrame(metrics_data)

# Set 'Metric' as the index to use it for the heatmap row labels
df_metrics.set_index('Metric', inplace=True)

# Labels for the metrics
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

target_values = [0.90, 0.85, 0.85, 0.85]

# Plotting Bullet Graph
plt.figure(figsize=(6,6))

# Define positions for the bullet graphs
y_positions = range(len(metrics_labels))

# Loop through each metric and plot a bullet graph for each model
for i, metric in enumerate(metrics_labels):
    # Target bar
    plt.barh(y_positions[i] + 0.2, target_values[i], height=0.4, color='lightgrey', label='Target' if i == 0 else "")

    # Logistic Regression bar
    plt.barh(y_positions[i] + 0.1, metrics_log_reg[i], height=0.2, color='blue', label='Logistic Regression' if i == 0 else "")

    # Random Forest bar
    plt.barh(y_positions[i], metrics_rf[i], height=0.2, color='black', label='Random Forest' if i == 0 else "")

    # SVM bar
    plt.barh(y_positions[i] - 0.1, metrics_svm[i], height=0.2, color='grey', label='SVM' if i == 0 else "")

# Add labels and title
plt.yticks(y_positions, metrics_labels)
plt.xlabel('Score')
plt.title('Bullet Graph of Model Performance (Accuracy, Precision, Recall, F1-Score)', fontweight='bold')

# Add legend
plt.legend()

# Show plot
plt.show()
