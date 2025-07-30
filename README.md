# FAKE-NEWS-DETECTOR-
📰 Fake News Detector: A machine learning-based Fake News Detector that classifies news as real or fake using TF-IDF and classical models like Logistic Regression. Trained with Stratified K-Fold validation

📄 Dataset
we used the WELFake_Dataset.csv availaible online [click here to get the Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)




📌 Features

- Text preprocessing and cleaning pipeline
- Vectorization using TF-IDF and CountVectorizer
- Trained multiple ML models: Logistic Regression, Naive Bayes
- Stratified K-Fold Cross Validation
- Lightweight design for local training

📁 Project Structure
- preprocessing.py #for text preprocessing
- vectorize.py # Vectorizer training and saving
- main.py # Model training and evaluation
- test_dataset.py #for testing the model on dataset
- test.py #for testing model on custom articles
- requirements.txt #modules needed

🧠 Model Flow
1. downlooad the dataset from this link [Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
2. run the preprocessing.py (it will create a file named cleaned_preprocessed_news.csv)
3. PS. this original dataset has ~9000 duplicate samples and articles length inconsistency make sure to fix that 
4. run vectorize.py
   
   it will create 3 files 
   - tfidfs_words.pkl
   - tfidfs_chars.pkl
   - count_vectorizer.pkl
5. run main.py
  The best model will be saved as best_model.pkl
6. run test_model.py (make sure to give it real news articles and its word limit is small make sure to give only a paragraph)

👨‍💻 Author
Made with ❤️ by Umer
For learning purposes and academic demonstration.


🔗LINKEDIN:
Feel free to connect with me on [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/umerjavaidx)



