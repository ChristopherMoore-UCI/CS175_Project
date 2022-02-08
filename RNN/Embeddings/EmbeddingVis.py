import gensim
from gensim.models import Word2Vec,KeyedVectors
import plotly
import numpy as np
import pandas as pd

def getEmbeddings(fileNames):
	embeddings = []
	for fn in fileNames:
		if ".bin" in fn:
			embeddings.append(KeyedVectors.load_word2vec_format(fn, binary=True))
		else:
			embeddings.append(KeyedVectors.load(fn))
		
	return embeddings


