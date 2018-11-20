import nltk
import os
import networkx as nx 
from nltk.tree import Tree
import networkx as nx
import matplotlib.pyplot as plt
import random
import _pickle as pickle

class DependencyParser:
    def __init__(self):
        from nltk.parse.stanford import StanfordDependencyParser
        from nltk.tokenize import PunktSentenceTokenizer, WordPunctTokenizer
        self.parser = StanfordDependencyParser()
        self.sentenceTokenizer = PunktSentenceTokenizer()
        self.wordTokenizer = WordPunctTokenizer()

    def parse(self, file):
        f = open(file).read()
        rainForest = {
            "sentence":[],
            "forest": []
        }
        sentences = self.sentenceTokenizer.tokenize(f)
        # print(sentences)
        segSentences = [self.wordTokenizer.tokenize(sentence) for sentence in sentences][0]
        # posTaggedSentences = [nl] 
        sentenceForest = []
        for i in range(len(sentences)):
            sentenceTree = [parse.tree() for parse in self.parser.raw_parse(sentences[i])][0]
            rainForest["sentence"].append(sentences[i])
            rainForest["forest"].append(sentenceTree)
        return rainForest
        
def main():
    TRAINDIR = "C:\\Users\\mdnur\\Projects\\Gram_ai\\data\\Abstracts_3400\\"
    TREEDIR = "C:\\Users\\mdnur\\Projects\\Gram_ai\\data\\Abstracts_3400\\trees"
    numFiles = len(os.listdir(TRAINDIR))
    chosen = random.randint(0,numFiles-1)
    os.environ["CLASSPATH"] = "C:\\Users\\mdnur\\Downloads\\Stanford\\stanford-corenlp-full-2018-10-05"
    p = DependencyParser()
    textfiles = [file for file in os.listdir(TRAINDIR) if not file.endswith(".iob")]
    rainforest = p.parse(os.path.join(TRAINDIR, os.listdir(TRAINDIR)[chosen]))
    f = open(os.path.join(TREEDIR,"sampletree.pickle"), "wb")
    print(rainforest)
    pickle.dump(rainforest, f)
    f.close()
    print("Dumped")

if __name__ == "__main__":
    main()
