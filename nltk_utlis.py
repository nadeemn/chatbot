import nltk
import numpy as np

from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
stemmer = PorterStemmer()
def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)

def stemming(word):
    """
    stemming = find the root form of the word
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenizsed_sentence, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    """
    tokenizsed_sentence = [stemming(w) for w in tokenizsed_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in tokenizsed_sentence:
            bag[index] = 1.0
    return bag
