import joblib
from scipy.sparse import hstack

# Optional: If your model was trained on preprocessed text, import your function
# from preprocess_module import clean_text_improved

# 📦 Load model and vectorizers
model = joblib.load("best_model.pkl")
tfidf_words = joblib.load("tfidf_words.pkl")
tfidf_chars = joblib.load("tfidf_chars.pkl")
count_vect = joblib.load("count_vectorizer.pkl")

print("\n🔍 Fake News Detector")
print("====================================")
print("📄 Enter your news article (paste text and press Enter twice):")

# 📝 Get user input
lines = []
while True:
    line = input()
    if line.strip() == "":
        break
    lines.append(line)
article = " ".join(lines)

# ✅ Preprocess the article if required
# article = clean_text_improved(article)

# ✨ Vectorize
word_features = tfidf_words.transform([article])
char_features = tfidf_chars.transform([article])
count_features = count_vect.transform([article])

# 🧱 Combine all feature types
features = hstack([word_features, char_features, count_features])

# 🔮 Predict
prediction = model.predict(features)[0]

# 📢 Show result
print("\n====================================")
print("📄 Your Input:\n", article[:500], "..." if len(article) > 500 else "")
print("🧠 Prediction:", "FAKE ❌" if prediction == 1 else "REAL ✅")
print("====================================\n")
