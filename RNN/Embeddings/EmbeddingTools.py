import gensim
from gensim.models import Word2Vec,KeyedVectors
import torch 
import torch.nn as nn


def getEmbedding(fileNames):
	
	isList = True

	if type(fileNames) != list:
		fileNames = [fileNames]
		isList = False

	embeddings = []
	for fn in fileNames:
		if ".bin" in fn:
			embeddings.append(KeyedVectors.load_word2vec_format(fn, binary=True))
		else:
			embeddings.append(KeyedVectors.load(fn))
	
	if isList:
		return embeddings
	else:
		return embeddings[0]



def embeddingLayerFromFile(fn):
	embedding = getEmbedding(fn)

	weights = torch.FloatTensor(embedding.wv.vectors)

	embeddingLayer = nn.Embedding.from_pretrained(weights)
	embeddingLayer.requires_grad = False

	return embeddingLayer

def embeddingLayerFromEmbedding(embedding):

	weights = torch.FloatTensor(embedding.wv.vectors)

	embeddingLayer = nn.Embedding.from_pretrained(weights)
	embeddingLayer.requires_grad = False

	return embeddingLayer



