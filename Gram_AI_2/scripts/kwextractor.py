import pandas as pd 
import nltk
import collections
from gensim import corpora
from gensim.corpora import Dictionary
# nltk.download('averaged_perceptron_tagger')

# def get_key(value, dict):
#     for k, v in dict.items

metadata = pd.read_csv("..\\data\\absrecord.csv")
print(len(metadata['filename'].values))
fullvocab = []

from preprocessor import preprocess, flatten

for record in range(len(metadata)):
    # print(100*record/len(metadata))
    fullvocab.append(preprocess(str(metadata.iloc[record]['body']))[0])
print(fullvocab)
maindict = Dictionary(fullvocab)
i = 0
fulldict = []
for document in fullvocab:
    temp = []
    print(100 * i/len(fullvocab))
    i += 1
    document = list(sorted(set(document)))
    for token in document:
        if token in list(maindict.values()):
            for key, value in list(maindict.items()):
                if token == value:
                    temp.append({"id":key, "name":token})
                    # print({"id":key, "name":token})
    fulldict.append(temp)

b = metadata['filename'].values
print(fulldict)
a = pd.DataFrame({'keywords': fulldict})
metadata.append(a)
metadata.to_csv("..\\data\\keywords.csv")