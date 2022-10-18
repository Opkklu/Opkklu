# ========================= READ ME =========================
# After running model.py script, it was time to test
# if the neural machine translation was capable of generalizing well on unseen data.
# So it was created a new dataset called test-dataset.txt with
# some text from Wikipedia. The code below cleans this test data
# and separates it into tokens of 27 characters each,
# KEY--- plus 21 letters of the test-dataset.txt
# Then the text is encrypted with letter B and the model gets
# the ciphers. The predictions are shown in the end.
# If you run this code on Google Colab, install keras_self_attention
# by running on a notebook:
# !pip install keras_self_attention

from keras.models import load_model
from keras_self_attention import SeqSelfAttention
from numpy import array, argmax

def load_data(filename):
  """It loads a text file, in this case test-dataset.txt"""
  with open(filename, "r", encoding = "utf-8", errors = "ignore") as file_txt:
    lines = file_txt.readlines()
    file_txt.close()
  return lines


def clean_text(data):
  """It cleans the text stored in a list and returns another list
     made of tokens (sentences) with 27 characters each."""
  counter = 0
  # Removing \n from each line
  while counter < len(data):
    data[counter] = data[counter].replace("\n", "").strip()
    counter += 1
  # Removing all special characters and numbers
  cleanData = list()
  token = ""
  for line in data:
    if line != "":
      for letter in line:
        if letter.isalpha():
          try:
            int(letter)
          except:
            if len(token) != 21:
              token += letter.upper()
            else:
              # KEY--- was added on training data to make the model to learn faster.
              # It must be added on test data to keep the integrity of the predictions,
              # as the model recognize this pattern on the sentences.
              token = "KEY---" + token 
              cleanData.append(token)
              token = ""

  return cleanData      


def caesar_encryption(data, key):
  """Encript tokens accoring to a key passed as the last parameter."""
  key = key.upper()
  if key not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    print(f"Invalid key: {key}")
    print("The key must be A, B, C ... X, Y, or Z")
    return None
  else:
    print("Encrypting data...")
    alphabet = {"-": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7,
                "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13,
                "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20,
                "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26}

    # Converting tokens to integers + key number
    plaintext = list()
    ciphers = list()
    errors = 0
    for sentence in data:
      text = list()
      cipher = list()
      c = 0
      while c < len(sentence):
        if c >= 0 and c <= 5:
          num = alphabet[sentence[c]]
          text.append(num)
          cipher.append(num)
        else:
          try:
            num = alphabet[sentence[c]]
            text.append(num)
            t = num + alphabet[key]
            if t > 26:
              t = t - 26
            cipher.append(t)
          except:
            text.append(alphabet["A"])
            cipher.append(alphabet["A"])
            errors += 1
        c += 1
      plaintext.append(text)
      ciphers.append(cipher)
    print("Finished")
    print(f"Errors: {errors}")
  
  return array(plaintext), array(ciphers)


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
          if counter == 27:
              words.append(word)
              word = ""
              counter = 0
  return words


filename = "drive/MyDrive/ciphers-lstm/caesar/data/test-dataset.txt"
data = load_data(filename)
cleanData = clean_text(data)
key = "B"
plaintext, ciphertext = caesar_encryption(cleanData, key)
plaintext = one_hot(plaintext, 27, "Plaintext") # vocabSize is the number of different letters on dataset

# Loading the model and making predictions
model = load_model("drive/MyDrive/ciphers-lstm/caesar/model-info/model.h5", custom_objects = {"SeqSelfAttention": SeqSelfAttention})
points = 0
letterPoints = 0
totalLetters = len(ciphertext) * 27
for i in range(len(ciphertext)):
  x = ciphertext[i].reshape(1, 27)
  y = plaintext[i].reshape(1, 27, 27)
  pred = model.predict(x, verbose = 0)
  inputData = decode(x)
  outputData = decode(y)
  prediction = decode(pred)
  print(f"\n=============== Example {i + 1} ==============")
  print(f"{'Ciphertext:':<13} {inputData[0]}")
  print(f"{'Plaintext:':<13} {outputData[0]}")
  print(f"{'Prediction:':<13} {prediction[0]}")
  if outputData[0] == prediction[0]:
    points += 1
  c = 0
  while c < len(outputData[0]):
    if outputData[0][c] == prediction[0][c]:
      letterPoints += 1
    c += 1
    
print("\n===============================")
print(f"Accuracy (full token): {(points / len(ciphertext)):.2%}")
print(f"Letter accuracy: {(letterPoints / totalLetters):.2%}")
print("===============================")
print("The model predicts full sentences with good accuracy,")
print("but if you look the plaintexts and predictions printed,")
print("you will notice that in the incorrect results the model predicts")
print("most of the letters correctly. So it is a pretty good machine translation for Caesar cipher.")
print("There is no need to apply deep learning to solve Caesar ciphers because it is")
print("an easy encryption system to break. However, this work aims to show to the community")
print("that neural machine translation is a great approach to decode ciphertexts.")
print("It can be generalized to Vigenere, Autokey, and so on.")
print("You just need to provide the appropriate data and choose the right hyperparameters.")