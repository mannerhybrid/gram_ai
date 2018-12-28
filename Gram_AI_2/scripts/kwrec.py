import os
import pandas as pd 
import numpy as np 
import tensorboard as board
from preprocessor import preprocess, remove_stopwords
from simmodel import vectorize
from ast import literal_eval
from gensim import corpora
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dense,Dot,Input, Embedding, Reshape, LSTM, Concatenate, Conv1D, Reshape
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


def recommender_model(vocab_size, emb_dim = 100):
	# create model
	emb = Embedding(input_dim=vocab_size, output_dim=emb_dim)

	x1 = Input(shape=(1,None), batch_shape=(1,None), name="title")
	print(x1.shape)
	e1 = emb(x1)
	print(e1.shape)
	h1 = LSTM(50,input_shape=(1,None), return_sequences=False, stateful=True)(e1)

	x2 = Input(shape=(1,None),batch_shape=(1,None), name="body")
	e2 = emb(x2)
	h2 = LSTM(50,input_shape=(1,None),  return_sequences=False, stateful=True)(e2)

	q = Input(shape=(1,None), batch_shape=(1,None), name="query")
	eq = emb(q)
	hQ = LSTM(50,input_shape=(1,None),  return_sequences=False, stateful=True)(eq)

	sim_title_query = Dot(axes=1)([h1, hQ])
	sim_body_query = Dot(axes=1)([h2, hQ])

	sim = Concatenate()([sim_title_query, sim_body_query])
	sim = Dense(1, activation='sigmoid')(sim)
	model = Model(inputs=[x1, x2, q], outputs=[sim])
	sgd = SGD()
	model.compile(loss='mean_squared_error',optimizer=sgd)
	print(model.summary())
	
	return model


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names

    #Return empty list in case of missing/malformed data
    return []

def main():
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
	x_train_keywords = metadata['keywords']

	y_train = np.ones([len(x_train_body)])
	maindict = corpora.Dictionary.load("..\\models\\maindict.dict")

	vectors_titles = [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in x_train_titles]
	vectors_bodies =  [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in x_train_body]
	vectors_keywords = [np.array(maindict.doc2idx(doc, unknown_word_index=len(maindict))) for doc in x_train_keywords]
	print(vectors_titles[0].shape)
	max_len = {k:v for k,v in zip(['title', 'body', 'keywords'],[int(round(np.mean([len(v2) for v2 in v]))) for v in [vectors_titles, vectors_bodies, vectors_keywords]])}
	
	print(max_len)
	# vectors_titles = [v for v in keras.preprocessing.sequence.pad_sequences(vectors_titles, maxlen=max_len['title'])]
	# vectors_bodies = [v for v in keras.preprocessing.sequence.pad_sequences(vectors_bodies, maxlen=max_len['body'])]
	# vectors_keywords = [v for v in keras.preprocessing.sequence.pad_sequences(vectors_keywords, maxlen=max_len['keywords'])]
	
	print(vectors_titles[0].shape)
	model = recommender_model(len(maindict)+1)
	history = model.fit([vectors_titles, vectors_bodies, vectors_keywords],y_train, epochs=10, batch_size=1, validation_split=0.2)
	model.evaluate()

	import datetime
	model.save("..\\models\\model_{}_{}.hdf5".format(datetime.date.day, datetime.date.month))

	print(history.history.keys())
	# "Loss"
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()


if __name__ == "__main__":
	main()

