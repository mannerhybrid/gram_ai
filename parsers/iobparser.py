import os
import re
import numpy as np
from random import randint
import shutil 

# class NounPhraseExtractor:
#     def __init__(self, text):
#         from nltk.tokenize import PunktSentenceTokenizer
#         from nltk.

class IOBFileTagger:
    def __init__(self):
        import os
        import re
        import numpy as np
        return None

    def tag(self, file):
        fileReader = open(file, "r+")
        fileLines = fileReader.read().split("\n")
        # print(fileLines[0].split("\t")[1])
        self.entities = [(line.split("\t")[0], line.split("\t")[-1])  for line in fileLines if bool(line.split("\t")[-1] != "O") & bool(line.split("\t")[-1] != '')]
        self.entPhrases = [e for e, _ in self.entities]
        self.entTags = [t for _, t in self.entities]
        self.nounPhrases = self.chunk()
        return self.nounPhrases

    def extractNP(self):
        # startSet = list(zip(*self.entZipTags)[0])
        numEnts = list(set(self.startPoints))
        phrases = []
        for j in numEnts:
            phrases = [self.entPhrases[k] for k in [i for i,x in enumerate(self.startPoints) if x == j]]
            nP =" ".join(phrases)
            phrases.append(nP)
        return phrases

    def chunk(self):
        # Identify NP Beginnings
        tags = self.entTags
        tagN = 0
        startPoints = []
        for tag in tags:
            if bool(re.match(r'B+\W+', tag)):
                tagN += 1
                startPoints.append(tagN)
            else:
                startPoints.append(tagN)
        
        self.startPoints = list(np.subtract(np.array(startPoints),1))
        nP = self.extractNP()
        if len(nP) == len(tags):
            return list(zip(nP, tags))
        return nP

def main():
    ABSDIR = "C:\\Users\\mdnur\\Projects\\Gram_ai\\data\\Abstracts_3400"
    tagger = IOBFileTagger()
    validFiles = [file for file in os.listdir(ABSDIR) if file.endswith("txt.iob")]
    numFiles = len(validFiles)
    sample = np.random.randint(0, high=numFiles-1, size=(50,))
    TRAINDIR = os.path.join(ABSDIR, "train")
    TESTDIR = os.path.join(ABSDIR, "test")
    train_test_proportion = (500/numFiles)
    for path in [TRAINDIR, TESTDIR]:
        if not os.path.exists(path):            
            os.makedirs(path)
    trainFiles = [validFiles[idx] for idx in sample]
    for file in trainFiles:
        shutil.move(os.path.join(ABSDIR, file), os.path.join(TRAINDIR, file))
    print("Files moved!")
    nounPhrases = [""]
    print(sample)
    for file in os.listdir(TRAINDIR):
        nounPhrases.extend(tagger.tag(os.path.join(TRAINDIR, file)))
    # nounPhrases.extend(tagger.chunk())
    print(set(nounPhrases))

if __name__ == "__main__":
    main()