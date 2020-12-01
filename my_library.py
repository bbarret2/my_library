import pandas as pd



def hello():
  print('hello')

def dead_week(number):
  return number

def process_bio(bio):
  good_words = []
  doc = nlp(bio)
  for i in range(len(doc)):
    token = doc[i]
    if token.is_alpha and not token.is_oov and not token.is_stop:
      good_words += [token.text]
  return good_words

def all_bayes(training_table, word_bag, bio):
  all_classes = word_bag.columns.to_list()  #does not include word column because it is index
  n = len(all_classes)
  results = []
  for i in range(n):
    c = all_classes[i]  #a class, e.g., 'C1'
    result = naive_bayes(training_table, word_bag, bio, c)
    results += [[result,c]]
  return sorted(results, reverse=True)

def word_by_class_probability(training_table, word_bag, word, a_class, laplace=1):
  class_list = training_table['Class'].to_list()
  d = len(set(class_list))
  class_count = class_list.count(a_class)  #number of bios of a_class
  word_count = word_bag.loc[word, a_class] if word in word_bag.index else 0 #bios of a_class that used the word
  return (word_count+laplace)/(class_count + d*laplace)

def naive_bayes(training_table, word_bag, bio, a_class):
  good_words = process_bio(bio)
  n = len(good_words)
  p_o=class_probability(training_table, a_class)
  numerator_list =[p_o]
  for i in range(n):
    word=good_words[i]
    p_word_class = word_by_class_probability(training_table,word_bag,word,a_class)
    numerator_list += [p_word_class]

  numerator = 1
  for number in numerator_list:
    numerator*= number
  import math
  numerator = 0
  for number in numerator_list:
    numerator += math.log(number)
  return(numerator)

def all_bayes(training_table, word_bag, bio):
  all_classes = word_bag.columns.to_list()  #does not include word column because it is index
  results = []
  for i in range(len(all_classes)):
    c = all_classes[i]
    result = naive_bayes(training_table, word_bag, bio, c)
    results += [[result,c]]
  return sorted(results, reverse=True)
