# ========================= READ ME =========================
# This script creates an encoder-decoder LSTM model
# (a neural machine translation) to learn the representation
# of the Vigenere cipher. In order words, the program trains
# a deep learning model to decipher Vigenere ciphers.
# The keys used to encrypt the database are among
# the 10,000 most used English words according to
# n-gram frequency analysis of the Google's Trillion word corpus.
# There is no need to run this code again because the model was saved
# in the directory /vigenere/model-info/model.h5

# If you want to test the deep learning application,
# go to translator.py under the folder /vigenere
# You will not need Colab Pro to execute translator.py
# If you execute this script on Google Colab,
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
  """It randomly shuffles and splits the data between plaintext and ciphertext."""
  print(f"Spliting dataset...")
  shuffle(data)
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
          if counter == 38:
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
    """resolver = tf.distribute.cluster_resolver.TPUClusterResolver(address) # TPU detector
    tf.config.experimental_connect_to_cluster(resolver)
    # Initialize the TPU
    tf.tpu.experimental.initialize_tpu_system(resolver)"""
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    """print(f"All devices: {tf.config.list_logical_devices('TPU')}")
    print(f"Number of devices: {len(tf.config.list_logical_devices('TPU'))}")"""
    strategy = tf.distribute.TPUStrategy(tpu) # define the distribution strategy
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
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of GPUs: {strategy.num_replicas_in_sync}")
    with strategy.scope():
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


# TPU will be used because it demonstrated to be faster for training than GPU
data = load_dataset("drive/MyDrive/ciphers-lstm/vigenere/data/final-dataset.txt")
plaintext, ciphertext = split_dataset(data)

# Cleaning up memory
# It is common to use almost all RAM memory when training a deep learning model
# That is why not needed variables are deleted below
del(data)
collect()

Xtrain = ciphertext
Ytrain = plaintext

del(plaintext, ciphertext)
collect()

Xtrain = to_integers(Xtrain, "Xtrain")
Ytrain = to_integers(Ytrain, "Ytrain")
Ytrain = one_hot(Ytrain, 27, "Ytrain") # the second parameter is the number of different symbols on dataset
  
# Define model
# The model was trained on Google Colab free edition,
# and its hyperparameters were found after lots of trial and error.
# 300 is the number of neurons
# If you want GPU, type GPU as the last parameter
model = define_model(27, 27, 38, 38, 300, "TPU")
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# Sumarize the model
print(model.summary())
plot_model(model, to_file = "drive/MyDrive/ciphers-lstm/vigenere/model-info/model.png", show_shapes = True)

# Fit the new model
checkpoint = ModelCheckpoint("drive/MyDrive/ciphers-lstm/vigenere/model-info/model.h5", monitor = "loss", verbose = 1, save_best_only = True, mode = "min")
# 20% of  the samples (40,000 tokens) will be using for testing
info = model.fit(Xtrain, Ytrain, epochs = 150, batch_size = 64, validation_split = 0.20, callbacks = [checkpoint], verbose = 1)

# Analyzing the model
model = load_model("drive/MyDrive/ciphers-lstm/vigenere/model-info/model.h5", custom_objects = {"SeqSelfAttention": SeqSelfAttention})
opt = tf.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])
      

# Cleaning up memory
del(Xtrain, Ytrain, checkpoint, opt)
collect()

print("========== Ploting the model performance (Vigenere cipher) ==========")
figure = plt.figure()
plt.plot(info.history["loss"])
plt.plot(info.history["val_loss"])
plt.title("Training loss vs validation loss (Vigenere cipher)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Training", "Validation"], loc = "upper right")
figure.savefig("drive/MyDrive/ciphers-lstm/vigenere/model-info/loss.png")
print("Performance saved")
plt.show()

# Saving model
modelArchitecture = model.to_json()
with open("drive/MyDrive/ciphers-lstm/vigenere/model-info/model-architecture.json", "wt") as json_file:
  json_file.write(modelArchitecture)
  json_file.close()
  print("Architecture saved")
model.save_weights("drive/MyDrive/ciphers-lstm/vigenere/model-info/model-weights.h5")
print("Weights saved")