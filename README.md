# chatbot-python
Chatbot with Deep Learning
This project is a simple chatbot implemented using TensorFlow and NLTK. The chatbot is trained on a set of predefined intents and responses, allowing it to recognize patterns in user input and respond appropriately.

Features
Pattern Recognition: Uses NLTK for tokenization and lemmatization of input patterns.
Deep Learning Model: Utilizes a deep learning model built with TensorFlow and Keras to classify user input into predefined intents.
Training Data Persistence: Saves processed words and classes into pickle files for later use.
Requirements
Python 3.6+
TensorFlow 2.0+
NLTK
NumPy
Setup and Installation
Clone the repository:

bash

git clone https://github.com/yourusername/chatbot.git
cd chatbot
Install the required libraries:

bash

pip install -r requirements.txt
Download NLTK resources:

python

import nltk
nltk.download('punkt')
nltk.download('wordnet')
Ensure the intents.json file is in the project directory. This file contains the training data for the chatbot.

Usage
Prepare the data:

The script processes the intents.json file to tokenize and lemmatize the patterns.
It generates a bag of words and corresponding output classes.
Processed words and classes are saved into words.pkl and classes.pkl respectively.
Train the model:

The deep learning model is defined using TensorFlow's Keras API.
The model is trained on the processed data for 200 epochs.
The trained model is saved as chatbot_model.keras.
Run the script:

bash
python chatbot.py

File Descriptions
chatbot.py: Main script for processing data and training the model.
intents.json: JSON file containing predefined intents and patterns.
words.pkl: Pickle file containing the list of words used for training.
classes.pkl: Pickle file containing the list of classes (intents) used for training.
chatbot_model.keras: Saved trained model.
