import json

from urllib import response
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

with open('intents.json') as file:
    data = json.load(file)

#features and training datas 
training_sentences = []
training_labels = []
labels=[]
response = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    response.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

num_classes = len(labels)

#let's explore sci-kit
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

#vectorize our  using Tokenizer
vocab_size = 1000
embedding_dim = 16 
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequence = tokenizer.texts_to_sequences(training_sentences)
padding_sequences = pad_sequences(sequence,truncating='post',maxlen=max_len)


#Model Training NEURAL NETWORK ARCHITECTURE
model = Sequential()
model.add(Embedding(vocab_size,embedding_dim,input_length=max_len))
model.add(GlobalAveragePooling1D())
#two neurons
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

#let's train our module
epochs = 500
history = model.fit(padding_sequences,np.array(training_labels),epochs=epochs)

#saving the trained model
model.save("chat_model")

#all the required file for future reference
import pickle

#save the filtered tokenizor
with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)

# to save the fitted label encoder
with open('label_encoder.pickle','wb') as ecn_file:
    pickle.dump(lbl_encoder,ecn_file,protocol=pickle.HIGHEST_PROTOCOL)
