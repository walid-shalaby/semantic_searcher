#spark-submit --verbose --master yarn --deploy-mode client --py-files ddoc2vecf.py movie_review.py
#spark-submit --verbose --master yarn --deploy-mode client gen_doc2vecR.py

from pyspark import SparkContext, SparkConf

conf = (SparkConf() \
    .set("spark.driver.maxResultSize", "2g"))

sc = SparkContext(conf=conf)

sc.addPyFile("boto.zip")
sc.addPyFile("smart_open.zip")
sc.addPyFile("gensim.zip")
sc.addPyFile("gensim-doc2vec-spark.zip")


from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

pos = sc.textFile("movie_review/positive").map(lambda s: (True, s.lower().split()))
neg = sc.textFile("movie_review/negative").map(lambda s: (False, s.lower().split()))
from ddoc2vec import DistDoc2Vec

data = (neg + pos).zipWithIndex().map(lambda (v, i): (i, v[0], v[1]))
sents = data.map(lambda (a,b,c): c)

model = Word2Vec(size=100, hs=0, negative=8)
dd2v = DistDoc2Vec(model, learn_hidden=False, num_partitions=5, num_iterations=10)
dd2v.build_vocab_from_rdd(sents, reset_hidden=False)
# train word2vec in driver
model.train(sents.collect())
model.save("review")
print "*** done training words ****"
print "*** len(model.vocab): %d ****" % len(model.vocab)
dd2v.train_sentences_cbow(data.map(lambda (i, l, v): TaggedDocument(words=v, tags=[i])))
dd2v.saveAsPickleFile("movie_review/docvectors")
