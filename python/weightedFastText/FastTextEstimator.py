from sklearn.base import ClassifierMixin,BaseEstimator
from weightedFastText import train_supervised,retrain_supervised
import numpy as np	
class FastTextEstimator(ClassifierMixin,BaseEstimator):
	def __init__(self,wordNgrams=1,minn=0,maxn=0,epoch=10,dim=100,verbose=0,pretrainedVectors=""):
		self.wordNgrams = wordNgrams
		self.minn = minn
		self.maxn = maxn
		self.epoch = epoch
		self.dim = dim
		self.verbose = verbose
		self.pretrainedVectors = pretrainedVectors
		super(ClassifierMixin, self).__init__()
	def fit(self,X,y,sample_weight = None,progressbar=None):
		import tempfile,os

		# Write Weights to Binary File for FastText
		import struct
		handleWeights = tempfile.NamedTemporaryFile(mode="wb",delete = False)	
		s = struct.pack('ll',len(X),1)
		handleWeights.write(s)
		if sample_weight is None:
			s = struct.pack('f'*len(X), *(len(X)*[1.0]))
		else:
			s = struct.pack('f'*len(X), *[len(X) * w for w in sample_weight])
		handleWeights.write(s)
		handleWeights.close()

		handleTrain = tempfile.NamedTemporaryFile(mode="w",delete = False)	
		traindocs = [x+" __label__"+str(y[i])+" __id__"+str(i) for i,x in enumerate(X)]
		from random import shuffle
		shuffle(traindocs)
		for d in traindocs:
			handleTrain.write(d+"\n")
		handleTrain.close()
		# handleTrial.close()
		# print(self.get_params())
		self._model = train_supervised(
			input  = handleTrain.name,
			weights = handleWeights.name,
			loss   = 'softmax',
			dim = self.dim,
			wordNgrams=self.wordNgrams,
			minn = self.minn,
			maxn = self.maxn,
			minCount = 0,
			epoch = self.epoch,
			verbose=self.verbose,
			pretrainedVectors = self.pretrainedVectors
		)
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
	def predict(self,X):
		predictions = self._model.predict(X)[0]
		return np.array([int(x[0][len("__label__"):]) for x in predictions])

	def predict_proba(self,X):
		predictions = self._model.predict(X,k=self.num_labels)[1]
		return predictions