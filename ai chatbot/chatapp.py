import streamlit as st
from PIL import Image
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os

# Set environment variable to turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load preprocessed data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

st.title("chatbot")

# Load bot logo image
bot_logo = Image.open("bot_logo.jpg")
bot_logo.thumbnail((70, 70))  # Resize image to half a centimeter

# Display bot logo
st.image(bot_logo, use_column_width=False)

# Define default message
default_message = "Hello, how can I assist you today?"

# Display default message
st.write(default_message)

def generate_response(user_input):
    intents_list = predict_class(user_input)
    response = get_response(intents_list, intents)
    return response

if 'generated' not in st.session_state: 
    st.session_state['generated'] = []

if 'past' not in st.session_state: 
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You:", "enter text here", key="input")
    return input_text

user_input = get_text()

if st.button("Send"):
    if user_input:
        output = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.image("bot_logo.jpg", caption="Bot", width=50)
        st.write(": " + st.session_state["generated"][i])
        st.write("You: " + st.session_state['past'][i])

