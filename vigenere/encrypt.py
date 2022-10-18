# This script creates the dataset that will be used
# to train the Vigenere encoder-decoder LSTM model.

from random import shuffle

def load_data(filename, clean):
  """It loads the raw dataset (a set of books from Gutenberg project).
     These books were already cleaned and separeted into tokens,
     phrases of 21 characters each. There are 2,174,580 tokens,
     but only 200,000 random samples were loaded each time.
     One of the reasons it won't use all dataset it's because
     the model was trained on a Google Colab Notebook,
     a cloud-computing application that runs Python code for AI projects.
     A free account was used, and it didn't have enough resources - 
     RAM and processing power - to train a model that incorporates all tokens."""
  print(f"Loading {filename}")
  with open(filename, "r", encoding = "utf-8", errors = "ignore") as file_txt:
    rawData = file_txt.readlines()
    file_txt.close()
  
  # Remove new line \n from each token
  counter = 0
  while counter < len(rawData):
    rawData[counter] = rawData[counter].replace("\n", "")
    counter += 1
  
  # Clean the keys file.
  # It removes the sentences with one letter
  # or those ones that contains the same character in all positions.
  if (clean): 
    cleanData = list()
    for sentence in rawData:
      if len(sentence) > 1:
        character = sentence[0]
        length = len(sentence)
        c = 0
        for letter in sentence:
          if letter == character:
            c += 1
        if length != c:
          cleanData.append(sentence.upper())
    
    maxLength = 0
    for sentence in cleanData:
      if len(sentence) > maxLength:
        maxLength = len(sentence)
  
    c = 0
    while c < len(cleanData):
      cleanData[c] = cleanData[c] + ("-" * ((maxLength + 1) - len(cleanData[c])))
      c += 1
    print("Finished")
    return cleanData
  else:
    print("Finished")
    shuffle(rawData)
    limit = 200000 # Define how many tokens the function will return
    rawData = rawData[:limit]
    return rawData


def vigenere(data, keys):
  """It gets a list of sentences and return a list of pairs.
     Each pair contains the plaintext (original string) and
     a ciphertext (a sentence encoded according to Vigenere cipher rules)."""
  # These keys are among the 10,000 most used English words,
  # according to a n-gram frequency analysis of the Google's Trillion word corpus.
  # Their length can vary from 2 to 16 letters.
  # The original list of keys can be found at:
  # https://github.com/first20hours/google-10000-english/blob/master/google-10000-english.txt
  
  # Each sentence has a total of characters equal to 21 + 1 + the greatest keyword.
  # In the first case, the bigger key has length 4.
  # So the tokens have 21 + 1 + 4 = 26 characters.
  # In the second case, the tokens have 30 characters.
  # In the third example, the tokens have 34 characters.
  # In the last case, the tokens have 38 characters.

  # As pointed on the paper Learning The Enigma With Recurrent Neural Networks,
  # the model learns faster and has more accuracy when it was trained with
  # pairs of sentences that have a part that does not change after encryption.
  # For example:
  # Key: AFGHANISTAN
  # Plaintext: AFGHANISTAN--DEEPLEARNINGISCHANGIN
  # Ciphertext: AFGHANISTAN--DJKWLRIJGIAGISHNHNTQF
  # The hashes "-" were ignored
  # This article is available on: https://www.arxiv-vanity.com/papers/1708.07576/

  alphabet = {"-": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6,
              "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13,
              "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20,
              "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26}

  reverseAlphabet = {0: "-", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F",
                     7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M",
                     14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T",
                     21: "U", 22: "V", 23: "W", 24: "X", 25: "Y", 26: "Z"}
  
  keysNumbers = list()
  for key in keys:
    nums = list()
    for letter in key:
      nums.append(alphabet[letter])
    keysNumbers.append(nums)
  
  errors = 0
  counter = 0
  key = 0
  limit = int(len(data) / len(keys))
  newData = list()
  for sequence in data:
    plaintextNumbers = list()
    for letter in sequence:
      try:
        plaintextNumbers.append(alphabet[letter])
      except:
        plaintextNumbers.append(alphabet["A"])
        errors += 1
    c = 0
    ciphertextNumbers = list()
    for num in plaintextNumbers:
      if num == 0:
        ciphertextNumbers.append(num)
      else:
        t = num + keysNumbers[key][c]
        if t > 26:
          t = t - 26
        ciphertextNumbers.append(t)
      if c == (len(keysNumbers[key]) - 1):
        c = 0
      else:
        c += 1
    cipher = ""
    for n in ciphertextNumbers:
      cipher += reverseAlphabet[n]
    keyChar = ""
    for n in keysNumbers[key]:
      keyChar += reverseAlphabet[n]
    newData.append([keyChar + cipher, keyChar + sequence])
    counter += 1
    if (counter == limit):
      if key < (len(keys) - 1):
        key += 1
      counter = 0
  print(f"Erros found {errors}")
  return newData


def save_dataset(filename, data):
  with open(filename, "w") as file_txt:
    for sentence in data:
      newSentence = sentence[0] + "\t" + sentence[1] + "\n"
      file_txt.write(newSentence)
    file_txt.close()
  print(f"{filename} was saved")


rawTexts = "drive/MyDrive/ciphers-lstm/vigenere/data/raw-data.txt"
keysFile = "drive/MyDrive/ciphers-lstm/vigenere/data/encryption-keys.txt"
data = load_data(rawTexts, False)
keys = load_data(keysFile, True)
print(f"Samples of data: {data[:3]}")
print(f"Samples of keys: {keys[:3]}")

newData = vigenere(data, keys)
print("Sample of coded/encoded data:")
for i in range(3):
  print(newData[i])
save_dataset("drive/MyDrive/ciphers-lstm/vigenere/data/final-dataset.txt", newData)