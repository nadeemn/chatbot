import nltk
import numpy as np

from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stemming(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenizsed_sentence, all_words):
    tokenizsed_sentence = [stemming(w) for w in tokenizsed_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in tokenizsed_sentence:
            bag[index] = 1.0
    return bag
