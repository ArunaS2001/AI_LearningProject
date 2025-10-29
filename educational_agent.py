# educational_agent.py
import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
with open('intents.json') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

# Prepare training data
X = []
y = []

tags = []
responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])
    responses[intent['tag']] = intent['responses']
    if intent['tag'] not in tags:
        tags.append(intent['tag'])

# Vectorize patterns
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X)

# Train classifier
model = LogisticRegression()
model.fit(X_train, y)

# Chat function
def chat():
    print("Educational Assistant: Hello! Ask me something (type 'quit' to exit).")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            print("Educational Assistant: Goodbye! Keep learning!")
            break
        inp_vector = vectorizer.transform([inp])
        tag_pred = model.predict(inp_vector)[0]
        response = random.choice(responses[tag_pred])
        print(f"Educational Assistant: {response}")

if __name__ == "__main__":
    chat()
