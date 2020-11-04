from sklearn.base import ClassifierMixin, BaseEstimator
from weightedFastText import train_supervised, retrain_supervised, load_model
import numpy as np


class FastTextEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self,
                 wordNgrams=1,
                 minn=0,
                 maxn=0,
                 epoch=10,
                 dim=100,
                 verbose=0,
                 pretrainedVectors="",
                 n_jobs=12):
        self.wordNgrams = wordNgrams
        self.minn = minn
        self.maxn = maxn
        self.epoch = epoch
        self.dim = dim
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pretrainedVectors = pretrainedVectors
        super(ClassifierMixin, self).__init__()

    def fit(self, X, y, sample_weight=None, progressbar=None):
        import tempfile, os

        # Write Weights to Binary File for FastText
        import struct
        handleWeights = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        s = struct.pack('ll', len(X), 1)
        handleWeights.write(s)
        if sample_weight is None:
            s = struct.pack('f' * len(X), *(len(X) * [1.0]))
        else:
            sample_weight = 1.0 * sample_weight / np.sum(sample_weight)
            print("NORMALIZED THE DAMN WEIGHTS!")
            s = struct.pack('f' * len(X), *[len(X) * w for w in sample_weight])
        handleWeights.write(s)
        handleWeights.close()

        handleTrain = tempfile.NamedTemporaryFile(mode="w",
                                                  delete=False,
                                                  encoding="utf-8")
        traindocs = [
            x.replace("\n", " ").strip() + " __label__" + str(y[i]) +
            " __id__" + str(i) for i, x in enumerate(X)
        ]
        from random import shuffle, seed
        seed(1)
        shuffle(traindocs)
        for d in traindocs:
            handleTrain.write(d + "\n")
        handleTrain.close()
        # handleTrial.close()
        # print(self.get_params())
        self._model = train_supervised(
            input=handleTrain.name,
            weights=handleWeights.name,
            loss='softmax',
            dim=self.dim,
            wordNgrams=self.wordNgrams,
            minn=self.minn,
            maxn=self.maxn,
            minCount=0,
            epoch=self.epoch,
            verbose=self.verbose,
            thread=self.n_jobs,
            pretrainedVectors=self.pretrainedVectors)
        # self.X = self._model.get_input_matrix()
        # right = self.X.transpose().dot(self.X)
        # s4squared,U = np.linalg.eig(right)
        # s4 = np.sqrt(s4squared)
        # print(s4)
        # init = self.X.dot(U.dot(np.diag(3.0/s4)).dot(U.transpose()))
        # print(len(self.X),self.X.shape)
        # for i in range(len(self.X)):
        # 	for j in range(self.dim):
        # 		self._model.set_input_at(i,j,init[i,j])

        # self.X = self._model.get_input_matrix()
        # right = self.X.transpose().dot(self.X)
        # s4squared,U = np.linalg.eig(right)
        # s4 = np.sqrt(s4squared)
        # # print(s4)

        # self._model = retrain_supervised(
        # 	self._model,
        # 	input  = handleTrain.name,
        # 	weights = handleWeights.name,
        # 	loss   = 'softmax',
        # 	dim = self.dim,
        # 	wordNgrams=self.wordNgrams,
        # 	minn = self.minn,
        # 	maxn = self.maxn,
        # 	epoch = self.epoch,
        # 	minCount  = 0,
        # 	verbose=self.verbose,
        # )
        self.num_labels = len(self._model.get_labels())
        os.remove(handleTrain.name)
        os.remove(handleWeights.name)
        if progressbar is not None:
            progressbar.update(1)

    def predict(self, X):
        predictions = self._model.predict(X)[0]
        return np.array([int(x[0][len("__label__"):]) for x in predictions])

    def predict_proba(self, X):
        predictions = self._model.predict(X, k=self.num_labels)
        # print(predictions[0][:10],predictions[1][:10])
        classes = np.array([[int(y[len("__label__"):]) for y in x]
                            for x in predictions[0]])
        order = np.argsort(classes, axis=1)
        # print(order[:10])
        return np.array([predictions[1][i, x] for i, x in enumerate(order)])

    def embed(self, X):
        return [self._model.get_sentence_vector(x) for x in X]

    def __getstate__(self):
        re = self.__dict__()
        del re["_model"]
        handleModel = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
        self._model.save_model(handleModel.name)
        handleModel.seek(0)
        re["_model"] = handleModel.read()
        print(re)
        return re

    # Make sure we can pickle this stuff
    def __getstate__(self):
        import tempfile, os
        re = self.__dict__
        handleModel = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
        self._model.save_model(handleModel.name)
        handleModel.seek(0)
        re["_model"] = handleModel.read()
        os.remove(handleModel.name)
        return re

    #make sure we can unpickle this stuff.
    def __setstate__(self, state):
        import tempfile, os
        handleModel = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
        handleModel.write(state["_model"])
        state["_model"] = load_model(handleModel.name)
        os.remove(handleModel.name)
        super(FastTextEstimator, self).__setstate__(state)
