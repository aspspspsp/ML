from nltk import word_tokenize
import nltk.data
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

a = 'abe abe abe abc e abc'
tokens = word_tokenize(a.lower())
print(tokens)