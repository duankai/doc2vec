import jieba
import re
from deepdist import DeepDist
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from pyspark.sql import HiveContext
from pyspark import SparkConf,SparkContext
from gensim import models

#spark = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value").getOrCreate()
appName = "Doc2Vec model training"
conf = SparkConf().setAppName(appName)
sc = SparkContext(conf = conf)
spark = HiveContext(sc)


regex=u"[\u4e00-\u9fa5]+"
p = re.compile(regex)


def split(jieba_list, iterator):
    sentences = []
    for i in iterator:
        seg_list = []
        #out_str = ""
        s = ""
        for c in i:
            if not c is None:
                s += c.encode('utf-8')
        id = s.split("__")[0]
        s = s.split("__")[1]
        wordList = jieba.cut(s, cut_all=False)
        for word in wordList:
            if word in jieba_list:
                continue
            if re.search(regex, word):
                seg_list.append(word)
        sentence = models.doc2vec.LabeledSentence(words=list(seg_list), tags=[id])
        sentences.append(sentence)
    return sentences

def gradient(model, sentence):  # executes on workers
    #syn0, doctag_syn0 = model.syn0.copy(), model.docvecs.doctag_syn0.copy()   # previous weights
    doctag_syn0 = model.docvecs.doctag_syn0.copy()
    model.train(sentence)
    return {'doctag_syn0': model.docvecs.doctag_syn0 - doctag_syn0}

def descent(model, update):      # executes on master
    #model.syn0 += update['syn0']
    model.docvecs.doctag_syn0 += update['doctag_syn0']

spark.sql("use jkgj_log")
df = spark.sql("SELECT concat(id,'__',description) FROM similar_info_regonize_data_source_final limit 100")

take = df.rdd.mapPartitions(lambda it: split([u'\u5fae\u8f6f',u'\u96c5\u9ed1',u'\u8f6c\u81ea'],it))

with DeepDist(Doc2Vec(take.collect(),alpha=.025, min_alpha=.025, min_count=1,workers=4, iter=4, size=500)) as dd:
    dd.train(take, gradient, descent)
    dd.model.save("doc2vec_model")
    print dd.model.docvecs.most_similar(51) 
