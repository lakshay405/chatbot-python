import random
import json
import pickle 
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers
import nltk
from nltk.stem import WordNetLemmatizer
import os

# Set environment variable to turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
try:
    with open('intents.json', 'r') as file:
        intents = json.load(file)
except FileNotFoundError:
    print("Error: 'intents.json' file not found.")
    exit()

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Process intents
for intent in intents.get('intents', []):
    for pattern in intent.get('patterns', []):
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save processed data
try:
    with open('words.pkl', 'wb') as file:
        pickle.dump(words, file)
    with open('classes.pkl', 'wb') as file:
        pickle.dump(classes, file)
except IOError:
    print("Error: Failed to save processed data.")
    exit()

training = []
outputEmpty = [0] * len(classes)

# Generate training data
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Build and train the model
model = tf.keras.Sequential([
    layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(trainY[0]), activation='softmax')
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

try:
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.keras')  # Save model using Keras format
    print('Model training completed successfully.')
except Exception as e:
    print(f"Error occurred during model training: {str(e)}")
