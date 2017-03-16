import codecs
from gensim import models
import copy
import sys
import time
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext

appName = "similar_info_calculate"
conf = SparkConf().setAppName(appName)
sc = SparkContext(conf = conf)
rdd = sc.textFile("dk/target_ids1.txt")

model = models.Doc2Vec.load("increase_model.doc2vec")
p1 = 0.65

def mapFunc(need_cal_id):     
    try:
        info_id = int(need_cal_id)
        similar_result = model.docvecs.most_similar(str(info_id), topn=20)
        similar_ids = [info_id]
        for (s_infoid, similarity) in similar_result:
            if similarity > p1:
                #print("> 0.75",s_infoid)
                similar_ids.append(int(s_infoid))
            else:
                break
        if len(similar_ids) > 1: 
            return str(similar_ids)+'\r\n'
        else:
            return "EMPTY\r\n"
    except:
        return "EMPTY\r\n"

result = rdd.repartition(1000).map(mapFunc)
#result.saveAsTextFile("test_result")
list_rs = result.collect()
#print(list_rs)

f = codecs.open("dk_test_file.txt",'w','utf-8',errors="ignore")
f.writelines([line for line in list_rs if line!="EMPTY\r\n"])

