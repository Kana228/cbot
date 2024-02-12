import json
import random
import pickle #its used for saving and loading trained models or data.
import numpy as np 
import nltk #The Natural Language Toolkit is a library for working with human language data. It provides tools for tasks such as tokenization, stemming, tagging, parsing
#nltk.download('punkt')
#nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer #process of reducing words to their base or root form -> do[doing,does,done]

#Tensorflow is an open-source machine learning library, keras is a high level neural networks that work on top of it
from tensorflow.keras.models import Sequential # is a linear stack of layers
from tensorflow.keras.layers import Dense,Activation,Dropout # dence- fully connected layer, activation- specify the activation function, dropout - regularization
from tensorflow.keras.optimizers import SGD #SGD - optimizer (Stochastic Gradient Descent) to train the neural network.
import tensorflow as tf
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',','(',')']

for intent in intents['intents']: #use hash maps
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #tokenize splits sentence in an individual words -> 'Hey, I am Kira' --> ['hey','I','am','Kira']
        words.extend(word_list) # use extend instead of append, as extend appends content to the list,append appends list to the list.
        documents.append((word_list,intent['tag'])) # append words that belong to some tag
        if intent['tag'] not in classes: # check if tag is in class
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words)) # set removes the duplicates and sorted returns it back to a list

classes = sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes) #template

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

train_x = training[:,:len(words)]
train_y = training[:,len(words):]


mdl = tf.keras.Sequential()
mdl.add(tf.keras.layers.Dense(128,input_shape=(len(train_x[0]),), activation='relu'))
mdl.add(tf.keras.layers.Dropout(0.5))
mdl.add(tf.keras.layers.Dense(64,activation='relu'))
mdl.add(tf.keras.layers.Dropout(0.5))
mdl.add(tf.keras.layers.Dense(len(train_y[0]),activation='softmax'))

sgd = tf.keras.optimizers.legacy.SGD(learning_rate = 0.01,momentum = 0.9,nesterov=True)
mdl.compile(loss = 'categorical_crossentropy',optimizer= sgd,metrics=['accuracy'])

hist = mdl.fit(train_x,train_y,epochs=200,batch_size=5,verbose = 1)
mdl.save('chatbot_model.keras',hist)

print('Done')

