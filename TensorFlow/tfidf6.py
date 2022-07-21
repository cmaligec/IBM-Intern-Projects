import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from numpy import dot
from numpy.linalg import norm
from collections import OrderedDict
import operator
from itertools import islice
import math
from codetiming import Timer
import io
import csv

#############################################################################
### Obtain the n-closest words in similarity to a given word.             ###
### Cluster words according to their level of similarity to a given word. ###
#############################################################################

### NOTE: PROVIDE PATH TO DOWNLOADED DATASET BELOW ###
corpus = pd.read_csv(r'C:\Users\CFSM\Desktop\CS4476\DATA\tfidf_sample_processed_IMBD.csv')
corpus = corpus.iloc[:,0].to_numpy()

# Turn the corpus text into a document-term sparse matrix of tfidf values.
vectorizer = TfidfVectorizer()
vectorized_corpus = vectorizer.fit_transform(corpus)

# Obtain the corpus vocabulary.
corpus_vocab = vectorizer.get_feature_names_out()
len_vocab = len(corpus_vocab)
print(f'\nThe number of vocabulary items in the sample: {len_vocab}\n')

# Apply Singular Value Decomposition (SVD), truncated version, to approximate the document-term sparse matrix in order to reduce its dimensionality and speed up processing time. The tf-idf document-term matrix (M) is decomposed into U*Sigma*V_transpose.
# If M is an mxn matrix, with m = num_documents and n = num_terms, then the three components have the following dimensions: U is mxr, Sigma is rxr, and V_transpose is rxn. The latent dimension, r = num_concepts where "concepts" group together like terms.

num_concepts = 20
lsa_obj = TruncatedSVD(n_components=num_concepts, n_iter=100, random_state=42)

# Reduce the dimensionality of the document-term matrix.
tfidf_lsa_data = lsa_obj.fit_transform(vectorized_corpus)
Sigma = lsa_obj.singular_values_

# Obtain concept-term matrix of shape num_concepts x len_vocab
V = lsa_obj.components_


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

#######################
### WORD EMBEDDINGS ###
#######################

out_v = io.open(r'C:\Users\CFSM\Desktop\CS4476\DATA\Embeddings\tfidf_vectors.tsv', 'w', encoding='utf-8')
out_m = io.open(r'C:\Users\CFSM\Desktop\CS4476\DATA\Embeddings\tfidf_metadata.tsv', 'w', encoding='utf-8')

for i in range(len_vocab):
    vec = word_tfidf(corpus_vocab[i])
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
for word in corpus_vocab:
  out_m.write(word + "\n")
out_v.close()
out_m.close()


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

print(f'Normal dot product of \"audience\" and \"mission\": {dot("audience", "mission")}')
print(f'Cubic kernel dot product of \"audience\" and \"mission\": {dot_cubic_kernel("audience", "mission")}\n')


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
    
print(f'Normal cosine similarity between \"audience\" and \"mission\": {sim_cos("audience", "mission")}')
print(f'Cubic kernel cosine similarity between \"audience\" and \"mission\": {sim_cos_cubic("audience", "mission")} \n')


#####################
### ACCURACY TEST ###
#####################

# See which of the two cosine similarities is more accurate by computing the similarity of a word to itself, which should have a value of exactly 1.0 plus or minus a small value of epsilon to take into account the limitations of float representation.
normal = 0
cubic = 0
len_corpus_vocab = len(corpus_vocab)
epsilon = 0.0000000000000003
for voc in corpus_vocab:
    norm_val = sim_cos(voc, voc)
    cubic_val = sim_cos_cubic(voc, voc)
    if(norm_val > 1.0 - epsilon and norm_val < 1.0 + epsilon):
        normal += 1
    if(cubic_val > 1.0 - epsilon and cubic_val < 1.0 + epsilon):
        cubic += 1
print(f'Normal cosine accuracy of sample: {round(100*normal/len_corpus_vocab, 2)}%')
print(f'Cubic kernel cosine accuracy of sample: {round(100*cubic/len_corpus_vocab, 2)}%\n')


#####################
### GET N-CLOSEST ###
#####################


def take(n, iterable):
    # Return first n items of the iterable as a list.
    return list(islice(iterable, n))


def get_sim_dict(word, sim_calc):
    # Return a dictionary of corpus vocabulary and corresponding similarity values to a given word parameter using a given simularity calculation (normal: sim_calc = sim_cos or kernel-based: sim_calc = sim_cos_cubic).
    sim_dict = {}
    if(word in corpus_vocab):
        for voc in corpus_vocab:
            sim_dict[voc] = sim_calc(word, voc)   
    else:
        for voc in corpus_vocab:
            sim_dict[voc] = 0
    return sim_dict


@Timer(name="decorator")
def n_closest(word, n, sim_calc):
    # Return the top n vocabulary closest to word (included) along with corresponding similarity values, calculated according to a given simularity calculation method (normal: sim_calc = sim_cos or kernel-based: sim_calc = sim_cos_cubic).
    sim_dict = get_sim_dict(word, sim_calc)
    sorted_dict = dict( sorted(sim_dict.items(), key=operator.itemgetter(1),reverse=True))
    n_closest = take(n, sorted_dict.items())
    return n_closest
    
print(f'10 closest vocabulary items for \"audience\" using normal cosine similarity: {n_closest("audience", 10, sim_cos)}\n')
print(f'10 closest vocabulary items for \"audience\" using cubic kernel cosine similarity: {n_closest("audience", 10, sim_cos_cubic)}\n')


###########################
### CLUSTERING OF TERMS ###
###########################


def get_sim(word, sim_calc):
    # For a given word parameter, return an array of similarity values  (corresponding in order to the vocabulary in corpus_vocab), calculated according to a given simularity calculation method (normal: sim_calc = sim_cos or kernel-based: sim_calc = sim_cos_cubic).
    sim_values = []
    if(word in corpus_vocab):
        for i in range(500): # For speed, only 500 vocab items are used.
            sim_values.append(sim_calc(word, corpus_vocab[i]))
    else:
        sim_values = [0]*len_corpus_vocab
    return sim_values

@Timer(name="decorator")
def get_clustering_data(word, level, sim_calc):
    # Cluster the vocabulary: Positives have similarity to a given word at or above threshold level. Similarity values are calculated according to a given simularity calculation method (normal: sim_calc = sim_cos or kernel-based: sim_calc = sim_cos_cubic).
    data = []
    for i in range(500): # For speed, only 500 vocab items are used.
        if(get_sim(word, sim_calc)[i] >= level):
            data.append([corpus_vocab[i], 1])
        else:
            data.append([corpus_vocab[i], 0])
    return data


normal_data = get_clustering_data("audience", 0.5, sim_cos)
cubic_kernel_data = get_clustering_data("audience", 0.5, sim_cos_cubic)

# Datafile for normal similarity clustering
header = ["word", "label"]
### NOTE: PROVIDE PATH TO DOWNLOAD DATASET BELOW ###
with open(r'C:\Users\CFSM\Desktop\CS4476\DATA\normal_clustered_data.csv', 'w', encoding='UTF8', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(normal_data)
print('Normal clustered datafile ready.')

# Datafile for cubic_kernel similarity clustering
header = ["word", "label"]
### NOTE: PROVIDE PATH TO DOWNLOAD DATASET BELOW ###
with open(r'C:\Users\CFSM\Desktop\CS4476\DATA\cubic_kernel_clustered_data.csv', 'w', encoding='UTF8', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(cubic_kernel_data)
print('Cubic kernel clustered datafile ready.')