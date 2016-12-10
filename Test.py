from nltk import word_tokenize
import nltk.data

a = 'abe abe abe abc e abc'
a = a.split()
a = list(set(a))
print(a)