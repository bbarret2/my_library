import pandas as pd
import pandas as pd
import numpy as np
import sys
import os
import math

#bring in spacy. Note might want to comment this out if don't like wait time.
os.system('python -m spacy download en_core_web_md')
import en_core_web_md
nlp = en_core_web_md.load()

#bring in uo-puddles
my_github_name = 'uo-puddles'
my_library_name = 'uo_puddles'
clone_url = f'git clone https://github.com/{my_github_name}/{my_library_name}.git'  #create the url to get the library
os.system(clone_url) # Cloning
import uo_puddles.uo_puddles as up

#put your functions below
def hello():
  print('hello')

def process_bio(bio):
  doc = nlp(bio)
  good_words = []

  for i in range(len(doc)):
    token = doc[i]

    if token.is_alpha and not token.is_oov and not token.is_stop:
      #looks good, add to good_tokens
      good_words += [token.text]

  return good_words

def class_probability(training_table, a_class):
  class_list = training_table['Class'].to_list()  #the Class column as a list
  class_count = class_list.count(a_class)
  return class_count/len(class_list)

def word_by_class_probability(training_table, word_bag, word, a_class, laplace=.1):
  class_list = training_table['Class'].to_list()
  d = len(set(class_list))
  class_count = class_list.count(a_class)  #number of bios of a_class
  word_count = word_bag.loc[word, a_class] if word in word_bag.index else 0 #bios of a_class that used the word
  return (word_count + laplace)/(class_count + laplace*d)

def naive_bayes(training_table, word_bag, bio, a_class):
  good_words = process_bio(bio)
  n = len(good_words)

  numerator_list = [class_probability(training_table, a_class)]  #start if off with P(O)
  for i in range(n):
    word = good_words[i]
    word_class = word_by_class_probability(training_table, word_bag, word, a_class)
    numerator_list += [word_class]

  numerator = 0
  for number in numerator_list:
    numerator += math.log(number)

  return numerator


def all_bayes(training_table, word_bag, bio):
  all_classes = word_bag.columns.to_list()  #does not include word column because it is index
  results = []
  for i in range(len(all_classes)):
    c = all_classes[i]
    result = naive_bayes(training_table, word_bag, bio, c)
    results += [[result,c]]
  return sorted(results, reverse=True)

