# Loads data (raw-dataset.txt) and encrypts it according to Caeser cipher rules

def load_data(filename):
  """It loads the raw dataset (a set of books from Gutenberg project).
     These books were already cleaned and separeted into tokens,
     phrases of 21 characters each. There are 2,174,580 tokens."""
  print(f"Loading {filename}")
  with open(filename, "r", encoding = "utf-8", errors = "ignore") as file_txt:
    rawData = file_txt.readlines()
    file_txt.close()
  
  # Remove new line \n from each token
  counter = 0
  while counter < len(rawData):
    rawData[counter] = rawData[counter].replace("\n", "")
    counter += 1
  print("Finished")
  return rawData


def caeser_encryption(data):
  """It gets a list of tokens and return a list of Caeser ciphers based on the input."""
  print("Encrypting data...")
  alphabet = {"-": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7,
              "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13,
              "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20,
              "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26}

  reverseAlphabet = {0: "-", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G",
                     8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M",
                     14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T",
                     21: "U", 22: "V", 23: "W", 24: "X", 25: "Y", 26: "Z"}

  limit = int(len(data) / len(alphabet))
  i = 0
  counter = 0
  errors = 0
  key = 1
  ciphers = list()
  while i < len(data):
    token = data[i]
    numbers = list()
    for letter in token:
      try:
        numbers.append(alphabet[letter])
      except:
        numbers.append(alphabet["A"])
        errors += 1
    cipherNumbers = list()
    for number in numbers:
      t = number + key
      if t > 26:
        t = t - 26
      cipherNumbers.append(t)
    cipher = ""
    for number in cipherNumbers:
      cipher += reverseAlphabet[number]
    ciphers.append("KEY---" + cipher)
    counter += 1
    i += 1
    if counter == limit:
      if key < (len(alphabet) - 1):
        key += 1
      counter = 0

  print(f"Errors: {errors}")
  print("Finished")
  return ciphers


def save_dataset(rawData, ciphers, filename):
  """It loads two list of tokens: one is the plaintext and the other is the ciphertext.
     Data is saved on directory specified in filename parameter."""
  print("Saving encrypted tokens...")
  if len(rawData) != len(ciphers):
    print("Length error")
    print("Plaintext and ciphertext must have the same numbers of tokens")
    return None
  else:
    with open(filename, "w") as file_txt:
      i = 0
      while i < len(rawData):
        sentence = ciphers[i][:6] + rawData[i] + "\t" + ciphers[i] + "\n"
        file_txt.write(sentence)
        i += 1
      file_txt.close()
  print("Finished")
  return None


filename = "drive/MyDrive/ciphers-lstm/caesar/data/raw-data.txt"
rawData = load_data(filename)
ciphers = caeser_encryption(rawData)
save_dataset(rawData, ciphers, "drive/MyDrive/ciphers-lstm/caesar/data/final-dataset.txt")