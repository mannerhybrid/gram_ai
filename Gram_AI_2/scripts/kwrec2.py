import os
import pandas as pd 
import numpy as np 
import tensorflow as tf
import tensorboard as board
from preprocessor import preprocess, remove_stopwords
from simmodel import vectorize
from ast import literal_eval
from gensim import corpora
import keras.backend as K
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dense,Dot,Input, LSTM, concatenate, Conv1D, Reshape
from keras.datasets import boston_housing
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train)

def permutate(keyword_list):
	from itertools import permutations
	names = list(permuations([i["name"] for i in keyword_list]))
	print(names)


def recommender_model(
    num_classes = 2,
    hidden_units = 100,
    batch_size=1,
    num_layers=1,
    learning_rate=0.001
):
    x_title = tf.placeholder(tf.float64, [batch_size, None], name='title_placeholder')
    x_body = tf.placeholder(tf.float64, [batch_size, None], name='body_placeholder')
    x_query = tf.placeholder(tf.float64, [batch_size, None], name='query_placeholder')

    seq_len_title = tf.placeholder(tf.int32, [batch_size])
    seq_len_body = tf.placeholder(tf.int32, [batch_size])
    seq_len_query = tf.placeholder(tf.int32, [batch_size])
    embeddings = tf.get_variable('embedding_matrix', [num_classes, hidden_units])
	
    h1_title = tf.nn.embedding_lookup(embeddings, x_title)
    h1_body = tf.nn.embedding_lookup(embeddings, x_body)
    h1_query = tf.nn.embedding_lookup(embeddings, x_query)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_units, state_is_tuple=True) # can use tf.nn.rnn_cell.GRUCell or tf.nn.rnn_cell.BasicRNNCell instead 
    cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
    init_state = cells.zero_state(batch_size, tf.float64)

    h2_title, final_h2_title = tf.nn.dynamic_rnn(cells, h1_title, initial_state=init_state, sequence_length=seq_len_title)
    h2_body, final_h2_body = tf.nn.dynamic_rnn(cells, h1_body, initial_state=init_state, sequence_length=seq_len_body)
    h2_query, final_h2_query = tf.nn.dynamic_rnn(cells, h1_query, initial_state=init_state, sequence_length=seq_len_query)

    with tf.variable_scope("similairty"):
        title_to_query = tf.tensordot(final_h2_title, final_h2_query, axes=0)
        body_to_query = tf.tensordot(final_h2_body, final_h2_query, axes=0)
        simvec = tf.concat(0, [title_to_query, body_to_query])

    with tf.variable_scope("softmax"):
        W = tf.get_variable('W', [2, 1])
        b = tf.get_variable('b', [1], initializer = tf.constant_initializer(0.0))
    
    logits = tf.matmul(simvec, W) + b
    

    return dict(
        x = dict(
            title=x_title,
            body=x_body,
            query=x_query
        ),
        seqlen=dict(
            title=seq_len_title,
            body=seq_len_body,
            query=seq_len_query
        ),
        init_state=init_state,
        final_states=dict(
            title=final_h2_title,
            body=final_h2_body,
            query=final_h2_query
        ),

    )


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names

    #Return empty list in case of missing/malformed data
    return []

seed = 7
os.system("cls")
np.random.seed(42)
p=0.75
absdir  = "..\\data"
metadata = pd.read_csv(os.path.join(absdir, "absrecord.csv"))

metadata['keywords'] = metadata['keywords'].apply(literal_eval)
metadata['keywords'] = metadata['keywords'].apply(get_list)
metadata['keywords'].replace('[]', np.nan, inplace=True)
metadata = metadata.dropna(subset=['keywords'])
x_train_titles = [preprocess(str(text))[0] for text in metadata['title'].values]
x_train_body = [preprocess(str(text))[0] for text in metadata['body'].values]
x_train_keywords = metadata['keywords'].values
y_train = np.ones([len(x_train_body)])
maindict = corpora.Dictionary.load("..\\models\\maindict.dict")

vectors_titles = [maindict.doc2idx(doc) for doc in x_train_titles]
vectors_bodies =  [maindict.doc2idx(doc) for doc in x_train_body]
vectors_keywords = [maindict.doc2idx(doc) for doc in x_train_keywords]

vocab_size = len(maindict)
print(vocab_size)
model = recommender_model(num_classes=vocab_size)



# # fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)
# # evaluate model with standardized dataset
