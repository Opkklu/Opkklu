# This script ran three times.
# On the first time, it selected 100 words that had 2 to 4 letters.
# The next time, it selected 100 words that had 5 to 8 letters.
# On the third time, it selected 100 words that had 9 to 12 letters.
# The last time, it selected 100 words that had 13 to 16 letters.
# The words picked up were used to encrypt part of the dataset
# that would be the training set. So four models were developed.

# It was found that it's better to build different models
# that learns to decipher Vigenere ciphers that were coded
# with different length keys, than to build just one model
# that tries to decipher text coded with any key.
# The one model approach would require millions of samples
# as well as a strong understanding about Attention for LSTM models. 

from random import shuffle

originalFile = "drive/MyDrive/ciphers-lstm/vigenere/data/google-10000-english-words.txt"
resultFile = "drive/MyDrive/ciphers-lstm/vigenere/data/encryption-keys.txt"

with open(originalFile, "r") as f:
  keys = f.readlines()

newKeys = list()
for key in keys:
  key = key.replace("\n", "")
  # Picking ups words from 13 to 16 letters
  if len(key) >= 13 and len(key) <= 16:
    newKeys.append(key)

with open(resultFile, "w") as f:
  # Just 100 random keys will be used for training
  shuffle(newKeys)
  print(f"A sample of new keys: {newKeys[:5]}")
  for key in newKeys[:100]:
    f.write("\n" + key)
  