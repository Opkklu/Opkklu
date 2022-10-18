# ========================= READ ME =========================
# This script creates an encoder-decoder LSTM model
# (a neural machine translation) to learn the representation
# of the Caeser cipher. In order words, the program trains
# a deep learning model to decipher Caesar ciphers.
# It was executed on Google Colab Pro (a paid subscription to access
# high quality CPU, GPU, and TPU to work with artificial intelligence).
# Google Colab has a free option, but it does not provide
# the computational resources needed to train this model.
# There is no need to run this code again because the model was saved
# in the directory /caesar/model-info
# If you want to test the deep learning application,
# go to translator.py under the folder /caesar
# You will not need Colab Pro to execute translator.py
# Last but not least, the neural network required a GPU and its training
# lasted for about 90 minutes. If you execute this script on Google Colab,
# know that the server stores all the libraries below except for keras_self_attention
# To install it run the following code on a notebook:
# !pip install keras_self_attention

import tensorflow as tf
from os import environ
from gc import collect
from random import shuffle
from numpy import argmax, array
from keras.utils.vis_utils import plot_model
from keras.models import Sequential, load_model
from keras_self_attention import SeqSelfAttention
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed, BatchNormalization
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

def load_dataset(filename):
  """It loads the dataset that will be used to train the model."""
  print(f"Loading dataset...")
  with open(filename, "r", encoding = "utf-8", errors = "ignore") as file_txt:
    data = file_txt.readlines()
    file_txt.close()
  
  # Removing new line \n at the end of each token
  i = 0
  while i < len(data):
    data[i] = data[i].replace("\n", "")
    i += 1
  print("Finished")
  return data


def split_dataset(data):
  """It shuffles and splits the data between plaintext and ciphertext."""
  # Randomly shuffle data, so even if a small part of the dataset is selected,
  # it will have high data variation
  shuffle(data)
  print(f"Spliting dataset...")
  plaintext, ciphertext = list(), list()
  for sentence in data:
    tokens = sentence.split("\t")
    plaintext.append(tokens[0])
    ciphertext.append(tokens[1])
  print("Finished")
  return plaintext, ciphertext


def to_integers(data, name):
  """Transform each letter into a integer according to the alphabet dict and stores it in a list."""
  alphabet = {"-": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7,
              "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13,
              "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20,
              "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26}
    
  errors = 0
  numbersSet = list()
  counter = 0
  print(f"Encoding {name} to integers")
  while (counter < len(data)):
      numbersWord = list()
      sentence = data[counter]
      for letter in sentence:
          try:
              numbersWord.append(alphabet[letter])
          except:
              letter = "A"
              numbersWord.append(alphabet[letter])
              errors += 1
      numbersSet.append(numbersWord)
      counter += 1
  print(f"Errors: {errors}")
  print("Finished")
  return array(numbersSet)


def one_hot(data, vocabSize, name):
  """One hot encodes a list of tokens."""
  print(f"One hot encoding {name}")
  dataVectors = list()
  for token in data:
    # vectors stores all vectors used to represent a token
    vectors = list()
    for num in token:
      # vector is a binary representation of one letter
      vector = [0 for i in range(vocabSize)]
      vector[num] = 1
      vectors.append(vector)
    dataVectors.append(vectors)
  
  dataVectors = array(dataVectors)
  print(f"dataVectors shape: {dataVectors.shape}")
  print("Finished")
  return dataVectors


def decode(data):
  """It gets one hot encoded vector and transforms them in their alphabetic representation."""
  reverseAlphabet = {0: "-", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G",
                     8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M",
                     14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T",
                     21: "U", 22: "V", 23: "W", 24: "X", 25: "Y", 26: "Z"}

  counter = 0
  word = ""
  words = list()
  for token in data:
      for row in token:
          if (len(data.shape)) == 3:
              num = argmax(row)
              word += reverseAlphabet[num]
          else:
              word += reverseAlphabet[row]
          counter += 1
          if counter == 21:
              words.append(word)
              word = ""
              counter = 0
  return words


def define_model(src, target, srcTimesteps, targetTimesteps, units, hardware):
  """Define NMT (neural machine translation) model."""
  hardware = hardware.lower()
  if (hardware == "tpu"):
    try:
        address = "grpc://" + environ["COLAB_TPU_ADDR"]
        print(f"Running on TPU {address}") # print TPU address
    except:
        print("No TPU found")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(address) # TPU detector
    tf.config.experimental_connect_to_cluster(resolver)
    # Initialize the TPU
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print(f"All devices: {tf.config.list_logical_devices('TPU')}")
    print(f"Number of devices: {len(tf.config.list_logical_devices('TPU'))}")
    strategy = tf.distribute.TPUStrategy(resolver) # define the distribution strategy
    with strategy.scope():
      # Encoder-Decoder
      model = Sequential()
      model.add(Embedding(src, units, input_length = srcTimesteps, mask_zero = True))
      model.add(LSTM(units, dropout = 0.20)) # dropout of 20% is applied to avoid overfitting
      model.add(RepeatVector(targetTimesteps))
      model.add(LSTM(units, dropout = 0.20, return_sequences = True))
      model.add(BatchNormalization())
      model.add(SeqSelfAttention()) # Because of the length of the tokens, it was added an attention layer
      model.add(TimeDistributed(Dense(target, activation = "softmax")))
  elif (hardware == "gpu"):
    # Encoder-Decoder  (same model definied on TPU)
    model = Sequential()
    model.add(Embedding(src, units, input_length = srcTimesteps, mask_zero = True))
    model.add(LSTM(units, dropout = 0.20))
    model.add(RepeatVector(targetTimesteps))
    model.add(LSTM(units, dropout = 0.20, return_sequences = True))
    model.add(BatchNormalization())
    model.add(SeqSelfAttention())
    model.add(TimeDistributed(Dense(target, activation = "softmax")))
  return model


filename = "drive/MyDrive/ciphers-lstm/caesar/data/final-dataset.txt"
data = load_dataset(filename)
plaintext, ciphertext = split_dataset(data)
# There are 2,174,580 tokens on final-dataset.txt
# The model was trained with 1,300,000 examples
# It was found that the TPU can train a model with about 300,000 samples
# So to train with more than one million tokens, the program got accessed to a GPU
numSamples = 1300000
limit = int(numSamples * 0.80) # limit is the amount of data used for training, 0.80 means 80%
Xtrain = ciphertext[:limit]
Ytrain = plaintext[:limit]
Xtest = ciphertext[limit:numSamples]
Ytest = plaintext[limit:numSamples]

# Cleaning up memory
# It is common to use almost all RAM memory when training a deep learning model.
# That is why not needed variables are deleted below
del(filename, data, plaintext, ciphertext, numSamples, limit)
collect()

Xtrain = to_integers(Xtrain, "Xtrain")
Ytrain = to_integers(Ytrain, "Ytrain")
Xtest = to_integers(Xtest, "Xtest")
Ytest = to_integers(Ytest, "Ytest")
Ytrain = one_hot(Ytrain, 27, "Ytrain") # vocabSize is the number of different symbols on dataset (length of the alphanet dictionary)
Ytest = one_hot(Ytest, 27, "Ytest")

# Define model
# The model was trained on Google Colab Pro GPU,
# and its hyperparameters were found after
# one month of trial and error.
model = define_model(27, 27, 27, 27, 256, "gpu") # if you want TPU, type tpu as the last parameter
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# Summarize defined model
print(model.summary())
plot_model(model, to_file = "drive/MyDrive/ciphers-lstm/caesar/model-info/model.png", show_shapes = True)

# Fit the model
filename = "drive/MyDrive/ciphers-lstm/caesar/model-info/model.h5"
checkpoint = ModelCheckpoint(filename, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")
info = model.fit(Xtrain, Ytrain, epochs = 10, batch_size = 64, validation_data = (Xtest, Ytest), callbacks = [checkpoint], verbose = 1)

# Now it saves the model architecture and weights in different files
modelArchitecture = model.to_json()
with open("drive/MyDrive/ciphers-lstm/caesar/model-info/model-architecture.json", "wt") as json_file:
  json_file.write(modelArchitecture)
  json_file.close()
model.save_weights("drive/MyDrive/ciphers-lstm/caesar/model-info/model-weights.h5")

# Evaluate the model
modelFinished = load_model("drive/MyDrive/ciphers-lstm/caesar/model-info/model.h5", custom_objects = {"SeqSelfAttention": SeqSelfAttention})
# Making some predictions
for i in range(4):
    newXtest = Xtest[i].reshape(1, 27)
    newyTest = Ytest[i].reshape(1, 27, 27)
    yPredict = modelFinished.predict(newXtest, verbose = 0)
    ciphertext = decode(newXtest)
    plaintext = decode(newyTest)
    prediction = decode(yPredict)
    print(f"{'Ciphertext:':<13} {ciphertext[0]}")
    print(f"{'Plaintext:':<13} {plaintext[0]}")
    print(f"{'Prediction:':<13} {prediction[0]}")

# Analyzing the model
print("========== Ploting the model performance (Caesar cipher) ==========")
figure = plt.figure()
plt.plot(info.history["loss"])
plt.plot(info.history["val_loss"])
plt.title("Training loss vs validation loss (Caesar cipher)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Training", "Validation"], loc = "upper right")
figure.savefig("drive/MyDrive/ciphers-lstm/caesar/model-info/loss.png")
plt.show()