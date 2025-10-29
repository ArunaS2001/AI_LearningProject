import streamlit as st
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents
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

# Streamlit interface
st.title("Educational Learning Assistant")
st.write("Hello! I am your AI learning assistant. Ask me anything (type 'quit' to exit).")

# Chat history
if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:")

if user_input:
    if user_input.lower() == "quit":
        st.write("Educational Assistant: Goodbye! Keep learning!")
    else:
        inp_vector = vectorizer.transform([user_input])
        tag_pred = model.predict(inp_vector)[0]
        bot_response = random.choice(responses[tag_pred])
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", bot_response))

# Display chat history
for speaker, message in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Educational Assistant:** {message}")
