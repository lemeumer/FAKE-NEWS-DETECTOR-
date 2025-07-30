import pandas as pd
import random
import joblib
from scipy.sparse import hstack

# 📦 Load trained model
model = joblib.load("best_model.pkl")

# 🧠 Load saved vectorizers
tfidf_words = joblib.load("tfidf_words.pkl")
tfidf_chars = joblib.load("tfidf_chars.pkl")
count_vect = joblib.load("count_vectorizer.pkl")

# 📄 Load test dataset
df = pd.read_csv("cleaned_preprocessed_news.csv")
df = df.dropna(subset=["text", "label"])  # Ensure relevant columns exist

# 🧪 Sample 10 random entries
samples = df.sample(10, random_state=42)

print("\n🧪 Testing 10 Random Samples from Dataset")
print("========================================\n")

correct = 0

for idx, row in samples.iterrows():
    text = row["text"]
    true_label = row["label"]

    # ✨ Vectorize using all three vectorizers
    word_features = tfidf_words.transform([text])
    char_features = tfidf_chars.transform([text])
    count_features = count_vect.transform([text])

    # 🧱 Combine features
    features = hstack([word_features, char_features, count_features])

    # 🔍 Make prediction
    prediction = model.predict(features)[0]

    # ✅ Accuracy check
    if prediction == true_label:
        correct += 1

    # 🖨️ Show output
    print(f"📰 Article: {text[:80]}...")
    print(f"🔍 Prediction: {'FAKE' if prediction == 1 else 'REAL'}")
    print(f"✅ Actual: {'FAKE' if true_label == 1 else 'REAL'}")
    print("--------------------------------------------------")

print(f"\n🎯 Accuracy on 10 samples: {correct}/10 correct ✅")
