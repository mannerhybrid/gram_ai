import tensorflow as tf 
import os, re
import nltk
from nltk.tokenize import WordPunctTokenizer, PunktSentenceTokenizer
from sklearn.model_selection import train_test_split, cross_val_score
import random
import gensim
from gensim import corpora
from gensim.corpora import Dictionary
import collections
from collections import defaultdict
import pandas as pd
import numpy as np 

outofvocab = []

def remove_stopwords(text):
    stopwords = [word.replace('\n', '') for word in open("..\\data\\stopwords.txt", "r").readlines()]
    text = ' '.join(w for w in text.split() if w not in stopwords and not len(w) == 1)
    return text

def preprocess(text):
    sent_tokenizer = PunktSentenceTokenizer()
    
    sentences = [sentence for sentence in sent_tokenizer.tokenize(str(text).lower())]
    sentences = [re.sub(r'\([\W\w\d\D%=.,]+\)', '', sentence) for sentence in sentences]
    sentences = [re.sub(r'\.', ' __PERIOD__ ', sentence) for sentence in sentences]
    sentences = [re.sub(r'-', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[\d\D]\.[\d\D]+', ' __DECIMAL__ ', sentence) for sentence in sentences]
    sentences = [re.sub(r' \d\D ', ' __NUMBER__ ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[;,:-@#%&\"\'�]', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[\(\)\[\]\+\*\/]', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[��]', ' ', sentence) for sentence in sentences]

    sentences = [remove_stopwords(sent) for sent in sentences]

    # sentences = " __END__ ".join(sentences)
    # sentences = sentences.split(" __END__ ")
    word_tokenizer = WordPunctTokenizer()
    lemmatizer = nltk.WordNetLemmatizer()
    # words1 = [word_tokenizer.tokenize(sent) for sent in sentences]
    texts = [gensim.corpora.textcorpus.remove_stopwords([lemmatizer.lemmatize(word) for word in word_tokenizer.tokenize(sent)]) for sent in sentences][0]
    
    frequency = defaultdict(int)
    # for sent in words1:
    #     for token in sent:
    #         frequency[token] += 1
    
    # texts = [[token for token in text if frequency[token] > 1] for text in texts]
    return texts

def flatten(l):
        if isinstance(l,(list,tuple)):
            if len(l):
                return flatten(l[0]) + flatten(l[1:])
            return []
        else:
            return [l]

def vectorize(docs, model=gensim.models.Word2Vec.load("..\\models\\word2vec.model")):
    oov = 0
    if isinstance(docs, list):
        return [vectorize(doc) for doc in docs]
    else:
        vectors = []
        for word in docs:
            try: 
                vector = model.wv[word]
                vectors.append(np.expand_dims(model.wv[word], 1))
            except ValueError:
                outofvocab.append(word)
                vectors.append(np.zeros([100, 1]))
                continue
            except KeyError: 
                outofvocab.append(word)
                vectors.append(np.zeros([100, 1]))
                continue
        # print([v.shape for v in vectors])
        try:
            vector =  np.concatenate(vectors, axis=1).transpose()
        except:
            vector = vectors[0].transpose()
        # print(vector.shape)
        return vector

    # print(outofvocab)

# def vectorize(docs, model=gensim.models.Word2Vec.load("..\\models\\word2vec.model")):
#     oov = 0
#     if isinstance(docs, list):
#         if len(docs) >= 1:
#             if isinstance(docs[0], list):
#                 return vectorize(docs[0]) + vectorize(docs[1:])
#             elif isinstance(docs[0], str):
#                 vectors = []
#                 for doc in docs:
#                     try:
#                         return model.wv[docs]
#                     except:
#                         return np.array([1,100])
#                 vectors = np.concatenate(vectors, axis=0)

#         else:
#             return []
        
def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def main():
    os.system("cls")
    np.random.seed(42)
    p=0.75
    absdir  = "..\\data\\"
    abscsv = pd.read_csv(os.path.join(absdir, "absrecord.csv"))
    
    x_train_titles = [preprocess(text) for text in abscsv.title.values]
    x_train_body = [preprocess(text) for text in abscsv.body.values]

    labels_train = np.ones([len(x_train_body)])
    alldocs = x_train_titles + x_train_body 
    
    dictionary = corpora.Dictionary(alldocs)
    corpus_titles  = [dictionary.doc2idx(text) for text in x_train_titles]
    corpus_bodies = [dictionary.doc2idx(text) for text in x_train_titles]

    vectors_titles = vectorize(x_train_titles)
    vectors_bodies = vectorize(x_train_body)
    seq_len_titles, seq_len_body = [[x.shape[0] for x in inp] for inp in [vectors_titles, vectors_bodies]]
    print([p.shape[0] for p in vectors_titles])
    print([p.shape[0] for p in vectors_bodies])
    vocab_size = len(corpora.Dictionary(alldocs).token2id) - len(set(outofvocab))

if __name__ == "__main__":
    main()