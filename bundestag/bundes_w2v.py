from gensim.models.word2vec import Word2Vec
import numpy as np


class BundesW2V():
    model = None
    vector_size = None
    window = None
    min_count = None
    sg = None
    workers = None

    def __init__(self, vector_size=50, window=8, min_count=1, sg=0, workers=46):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.workers = workers

    def init_model(self, X):
        self.model = Word2Vec(  X,
                                vector_size=self.vector_size,
                                window=self.window,
                                min_count=self.min_count,
                                sg=self.sg,
                                workers=self.workers)

    def embedding(self, X_list):
        # Function to convert a sentence (list of words) into a matrix representing the words in the embedding space
        def embed_sentence(sentence):
            embedded_sentence = []
            for word in sentence:
                if word in self.model.wv:
                    embedded_sentence.append(self.model.wv[word])

            return np.array(embedded_sentence)

        embed = []

        # Clarify if we really need this, comment out for now
        # X_list = [x if type(x) == str else '' for x in X_list]

        for sentence in X_list:
            embedded_sentence = embed_sentence(sentence)
            embed.append(embedded_sentence)

        return embed

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = Word2Vec.load(path)
