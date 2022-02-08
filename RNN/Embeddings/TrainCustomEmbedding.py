import gensim
from gensim.models import Word2Vec,KeyedVectors

class TrainIter:
	def __init__(self, fn):
			self.fn = fn

	def __iter__(self):
		with open(self.fn) as f:
			for line in f:
				yield line.strip().split(',')

def train(fname, epochs = 5):
	print("Training")
	itr = TrainIter(fname)
	model_cbow = Word2Vec(size=300, min_count=1, workers=4)
	model_cbow.build_vocab(itr)
	model_cbow.train(itr, total_examples=model.corpus_count, epochs=epochs)
	return model_cbow

