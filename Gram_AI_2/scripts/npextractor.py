import nltk
from nltk import RegexpParser
from preprocessor import preprocess

def extractNounPhrases(text):
    #    pattern = "NP: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}"
       pattern = "NP: {(<JJ>* <NN.*>+)?}"
       st = nltk.tokenize.PunktSentenceTokenizer()
       wt = nltk.tokenize.WordPunctTokenizer()
       
       sentences = st.tokenize(text)
       sentences = [preprocess(text) for text in sentences]
       sentences = [wt.tokenize(sent.lower()) for sent in sentences]
       sentences = [nltk.pos_tag(sent) for sent in sentences]
       cp = RegexpParser(pattern)

       nphrases_list = [[' '.join(leaf[0] for leaf in tree.leaves()) 
                            for tree in cp.parse(sent).subtrees() 
                            if tree.label()=='NP'] for sent in sentences]

       return nphrases_list

