# NLP-Assignement
Obiettivo: Classificare "geografico" vs. "non geografico"
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
import random
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
# Corpus di esempio
english_texts = ["This is a text in English.", "Another example of English content.", "This text is clearly in English."]
non_english_texts = ["Questo è un testo in italiano.", "Ceci est un texte en français.", "Dies ist ein Text auf Deutsch."]

# Creazione di etichette
texts = english_texts + non_english_texts
labels = ['ENGLISH'] * len(english_texts) + ['NON-ENGLISH'] * len(non_english_texts)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([word for word in tokens if word.isalnum() and word not in stop_words])

# Preprocessing dei testi
texts = [preprocess_text(text) for text in texts]
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)
y_pred = classifier.predict(X_test_vectorized)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Metriche
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='ENGLISH')
recall = recall_score(y_test, y_pred, pos_label='ENGLISH')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
