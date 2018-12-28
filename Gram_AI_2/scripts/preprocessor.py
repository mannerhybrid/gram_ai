from gensim.test.utils import common_texts, get_tmpfile
import gensim
from gensim.models import Word2Vec
from gensim import corpora, models
from gensim.models.keyedvectors import Doc2VecKeyedVectors
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
# from npextractor import extractNounPhrases 
import os, re
import nltk
from nltk.tokenize import WordPunctTokenizer, PunktSentenceTokenizer
from collections import Counter, defaultdict

def remove_stopwords(text):
    stopwords = [word.replace('\n', '') for word in open("..\\data\\stopwords.txt", "r").readlines()]
    text = ' '.join(w for w in text.split() if w not in stopwords and not len(w) == 1)
    return text

def flatten(l):
        if isinstance(l,(list,tuple)):
            if len(l):
                return flatten(l[0]) + flatten(l[1:])
            return []
        else:
            return [l]

def preprocess(text):
    sent_tokenizer = PunktSentenceTokenizer()
    
    sentences = [sentence.lower() for sentence in sent_tokenizer.tokenize(text)]
    sentences = [re.sub(r'\([\W\w\d\D%=.,]+\)', '', sentence) for sentence in sentences]
    sentences = [re.sub(r'\.', '', sentence) for sentence in sentences]
    sentences = [re.sub(r'-', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[\d\D]\.[\d\D]+', '', sentence) for sentence in sentences]
    sentences = [re.sub(r' \d\D ', '', sentence) for sentence in sentences]
    sentences = [re.sub(r'[;,:-@#%&\"\'�]', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[\(\)\[\]\+\*\/]', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'[��]', ' ', sentence) for sentence in sentences]

    sentences = [remove_stopwords(sent) for sent in sentences]

    sentences = " __END__ ".join(sentences)
    sentences = sentences.split(" __END__ ")
    word_tokenizer = WordPunctTokenizer()
    lemmatizer = nltk.WordNetLemmatizer()
    # words1 = [word_tokenizer.tokenize(sent) for sent in sentences]
    texts = [gensim.corpora.textcorpus.remove_stopwords([lemmatizer.lemmatize(word) for word in word_tokenizer.tokenize(sent)]) for sent in sentences]
    frequency = defaultdict(int)
    # for sent in texts:
    #     for token in sent:
    #         frequency[token] += 1
    
    # texts = [[token for token in text if frequency[token] > 1] for text in texts]
    return texts
    # return sentences

def tokenize(sentences):
    sentences = sentences.split(" __END__ ")
    word_tokenizer = WordPunctTokenizer()
    lemmatizer = nltk.WordNetLemmatizer()
    # words1 = [word_tokenizer.tokenize(sent) for sent in sentences]
    texts = [gensim.corpora.textcorpus.remove_stopwords([lemmatizer.lemmatize(word) for word in word_tokenizer.tokenize(sent)]) for sent in sentences]
    frequency = defaultdict(int)
    # for sent in words1:
    #     for token in sent:
    #         frequency[token] += 1
    
    # texts = [[token for token in text if frequency[token] > 1] for text in words1]
    return texts

def main():
    ABS = "C:\\Users\\mdnur\\Projects\\Gram_ai\\data\\abstracts"
    abstracts = [os.path.join(ABS, file) for file in os.listdir(ABS)]
    print(len(abstracts))
    texts = [str(open(file,"rb").read()) for file in abstracts]
    words = []
    for text in texts:
        wt = WordPunctTokenizer()
        words.extend(wt.tokenize(text.lower()))
    preprocessed_texts = [preprocess(text) for text in texts]
    preprocessed_texts = [tokenize(text) for text in preprocessed_texts]
    alldocs = []
    for i in preprocessed_texts:
        doc1 = []
        for i2 in i:
            doc1.extend(i2)
        alldocs.append(doc1)

    import numpy as np
    dictionary = corpora.Dictionary(alldocs)
    dictionary.save('.\\maindict.dict')
    vocabulary = sorted(dictionary.token2id.keys())

    corpus = [dictionary.doc2bow(text) for text in alldocs]
    corpora.MmCorpus.serialize(".\\hakimcorpus.mm", corpus)

    print(alldocs[0])

    # d2v = [TaggedDocument(doc, [i]) for i, doc in enumerate(alldocs)]
    # model = Doc2Vec(d2v)
    # model.train(d2v, total_examples=len(dictionary), epochs=20)
    # model.save("doc2vec.model")

    model2 = Word2Vec(alldocs)
    model2.train(alldocs, total_examples=len(alldocs), epochs=100)
    model2.save("word2vec.model")

    

if __name__ == "__main__":
    main()