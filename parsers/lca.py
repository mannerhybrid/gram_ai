import _pickle as pickle
import os
import nltk 
from nltk.tree import Tree
from iobparser import IOBFileTagger

class LeastCommonAncestor:
    import _pickle as pickle
    import os
    import nltk 
    from nltk.tree import Tree
    from iobparser import IOBFileTagger
    def __init__(self):
        return None
    
    def get_position(self, ent):
        return self.positions[self.entities.index(ent)]
    
    def build_path(self, ent):
        position = self.get_position(ent)
        path = []
        for i in range(len(position)):
            if i == 0:
                p = position
            else:
                p = p[:-1]
                p = tuple(p)
            path.append(p)
            p = list(p)
        path.append(())
        return path

    def lca(self, tree, ent1, ent2):
        self.positions = tree.treepositions()
        entities = []
        for position in self.positions:
            if type(tree[position]) == Tree:
                entities.append(tree[position].label())
            else:
                entities.append(tree[position])
        self.entities = entities

        path1 = self.build_path(ent1)
        path2 = self.build_path(ent2)

        intersection = list(set(path1).intersection(path2))
        ancestor = tree[intersection[min([len(inside) for inside in intersection])]].label()
        path1 = path1[:path1.index(intersection[min([len(inside) for inside in intersection])])+1]
        path2 = path2[:path2.index(intersection[min([len(inside) for inside in intersection])])+1]
        
        return ancestor, path1, path2

def main():
    TREEDIR = "C:\\Users\\mdnur\\Projects\\Gram_ai\\data\\Abstracts_3400\\trees"
    p = os.path.join(TREEDIR, "sampletree.pickle")
    forest = open(p, "rb")
    rainF = pickle.load(forest)
    sents = rainF["sentence"]; forests = rainF["forest"]
    lcaMain = LeastCommonAncestor()
    from random import randint
    i = forests[9]
    i.draw()
    print(forests.index(i))
    anc, p1, p2 = lcaMain.lca(i, "age", "obesity")
    path1 = [i[step].label() if type(i[step]) == Tree else i[step] for step in p1 ]
    path2 = [i[step].label()if type(i[step]) == Tree else i[step] for step in p2 ]
    print(anc, path1, path2)
    

if __name__ == "__main__":
    main()