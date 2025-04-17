import numpy as np
import re

def tokenize(sentence):
    """
    Tokenize a sentence using regex instead of nltk.
    """
    return re.findall(r'\b\w+\b', sentence.lower())

def stem(word):
    """
    Very simple stemmer: removes common suffixes.
    """
    suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words vector.
    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
