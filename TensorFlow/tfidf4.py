import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
import io
import csv
from collections import OrderedDict
import operator
from itertools import islice
import math
from codetiming import Timer
from sklearn.decomposition import TruncatedSVD

### Classify words according to their level of similarity to a given word.
### Obtain the n-closest words in similarity to a given word.


# Corpus retrieval from totality of IMDB Dataset
corpus = pd.read_csv(r'C:\Users\CFSM\Desktop\CS4476\DATA\tfidf_processed_IMBD.csv')
total_corpus = corpus.iloc[:,0].to_numpy()

sample_corpus_length = 1000
sample_corpus = corpus.iloc[:sample_corpus_length,0].to_numpy()

# CORPUS SELECTION #
corpus = sample_corpus
#corpus = total_corpus


# Get the tf-idf values for document-term sparse matrix.
vectorizer = TfidfVectorizer()
vectorized_corpus = vectorizer.fit_transform(corpus)

# Obtain the corpus vocabulary.
corpus_vocab = vectorizer.get_feature_names_out()
len_vocab = len(corpus_vocab)

# Apply Singular Value Decomposition (SVD), truncated version, to approximate the document-term sparse matrix in order to reduce its dimensionality and speed up processing time. The tf-idf document-term matrix (M) is decomposed into U*Sigma*V_transpose.
# If M is an mxn matrix, with m = num_documents and n = num_terms, then the three components have the following dimensions: U is mxr, Sigma is rxr, and V_transpose is rxn. The latent dimension, r = num_concepts where "concepts" group together like terms.

num_concepts = 100 # Select number of concepts desired.
lsa_obj = TruncatedSVD(n_components=num_concepts, n_iter=100, random_state=42)

# Reduce the dimensionality of the document-term matrix.
tfidf_lsa_data = lsa_obj.fit_transform(vectorized_corpus)
Sigma = lsa_obj.singular_values_

# Obtain concept-term matrix of shape num_concepts x len_vocab
V = lsa_obj.components_

#print(V.shape)


def word_index(word):
    # Get column index of concept-term matrix corresponding to desired word.
    for i in range(len_vocab):
        if(corpus_vocab[i] == word):
            return i

def word_tfidf(word):
    # From the concept-term matrix, get an array of tf-idf values corresponding to a given word.
    word_tfidf = []
    index = word_index(word)
    for i in range(num_concepts):
        word_tfidf.append(V[i, index])
    return word_tfidf


### DOT PRODUCTS ###
  
def dot(word, word2):
    x = word_tfidf(word)
    z = word_tfidf(word2)
    return np.matmul(x, z)

def dot_cubic_kernel(word, word2):
    # The cubic polynomial kernel dot product.
    x = word_tfidf(word)
    z = word_tfidf(word2)   
    return (1 + np.matmul(x,z))**3

print(f'Normal dot product is: {dot("audience", "mission")}')
print(f'Cubic kernel dot product is: {dot_cubic_kernel("audience", "mission")}')


### COSINE SIMILARITY ###
# Calculate the similarity of a word in the corpus with every word in the corpus, including itself.
# There are two types of similarity calculations: cosine similarity (sim_cos) and cosine similarity using the cubic polynomial kernel dot product (sim_cos_cubic).

def sim_cos(word, voc):
    dot1 = dot(word, voc)
    norm_wd = math.sqrt(dot(word, word))
    norm_voc = math.sqrt(dot(voc, voc))
    return dot1/(norm_wd*norm_voc)
    
def sim_cos_cubic(word, voc):
    dot1 = dot_cubic_kernel(word, voc)
    norm_wd = math.sqrt(dot_cubic_kernel(word, word))
    norm_voc = math.sqrt(dot_cubic_kernel(voc, voc))
    return dot1/(norm_wd*norm_voc)
    
print(f'Normal cosine is: {sim_cos("audience", "mission")}')
print(f'Cubic kernel cosine is: {sim_cos_cubic("audience", "mission")}')

### ACCURACY TEST ###
# See which of the two cosine similarities is more accurate by computing the similarity of a word to itself, which should have a value of exactly 1.0.
normal = 0
cubic = 0
for voc in corpus_vocab:
    if(sim_cos(voc, voc) != 1.0):
        normal += 1
    if(sim_cos_cubic(voc, voc) != 1.0):
        cubic += 1
print(f'Normal cosine inaccuracy: {normal} out of {len(corpus_vocab)}')
print(f'Cubic kernel cosine inaccuracy: {cubic} out of {len(corpus_vocab)}')


#####################
### GET N-CLOSEST ###
#####################

# SELECTION of type of similarity calculation #
sim_calc = sim_cos
#sim_calc = sim_cos_cubic

def take(n, iterable):
    # Return first n items of the iterable as a list.
    return list(islice(iterable, n))

@Timer(name="decorator")
def get_sim_dict(word):
    # Return a dictionary of corpus vocabulary and 
    # corresponding similarity values to a given word parameter.
    sim_dict = {}
    if(word in corpus_vocab):
        for voc in corpus_vocab:
            sim_dict[voc] = sim_calc(word, voc)   
    else:
        for voc in corpus_vocab:
            sim_dict[voc] = 0
    return sim_dict
#print(get_sim_dict("audience"))

@Timer(name="decorator")
def n_closest(word, n):
    # Return the top n vocabulary closest to word (included) 
    # along with corresponding similarity values.
    sim_dict = get_sim_dict(word)
    sorted_dict = dict( sorted(sim_dict.items(), key=operator.itemgetter(1),reverse=True))
    n_closest = take(n, sorted_dict.items())
    return n_closest
    
#print(f'10_closest for audience : {n_closest("audience", 10)}')


###############################
### CLASSIFICATION OF TERMS ###
###############################

# SELECTION of type of similarity calculation #
sim_calc = sim_cos
#sim_calc = sim_cos_cubic

@Timer(name="decorator")
def get_sim(word):
    # Return an array of similarity values to a given word parameter 
    # corresponding in order to the vocabulary in corpus_vocab.
    sim_values = []
    if(word in corpus_vocab):
        for voc in corpus_vocab:
            sim_values.append(sim_calc(word, voc))
    else:
        sim_values = [0]*len(corpus_vocab)
    return sim_values
#print(get_sim("audience"))

@Timer(name="decorator")
def get_classifier_data(word, level):
    # Classify the vocabulary: Positives have similarity to a given word at or above threshold level.
    data = []
    for i in range(len(corpus_vocab)):
        if(get_sim(word)[i] >= level):
            data.append([corpus_vocab[i], 1])
        else:
            data.append([corpus_vocab[i], 0])
    return data


#data = get_classifier_data("audience", 0.5)

# Datafile for similarity classifier
#header = ["word", "label"]
#with open(r"C:\Users\CFSM\Desktop\CS4476\CODE\tfidf\classified_data.csv", 'w', encoding='UTF8', newline = '') as f:
#    writer = csv.writer(f)
#    writer.writerow(header)
#    writer.writerows(data)
