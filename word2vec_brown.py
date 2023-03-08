import gensim
import logging
from nltk.corpus import brown

model = gensim.models.Word2Vec.load("./tmp/brown_model/word2vec.model")
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = brown.sents()
dim = [25, 100, 300]
wind = [2, 7, 15]
neg = [1, 7, 20]
model = gensim.models.Word2Vec(
    sentences,
    min_count=1,
    vector_size=25,
    window=2,
    negative=1,  # negatove>0 -> negative sampling
    hs=0,  # hs=0
    epochs=1,
    sg=1,  # sg=1 ->skip-gram
)
model.save("./tmp/brown_model/word2vec.model")

# model.wv.evaluate_word_analogies("./evaluation/win353.csv")

model.wv.evaluate_word_pairs('./evaluation/win353.csv')
