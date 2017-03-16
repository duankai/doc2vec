#!usr/bin/python
# -*- coding: utf-8 -*

import logging
import traceback
import codecs
from impala.dbapi import connect
from impala.util import as_pandas
import pandas as pd
import datetime
import time
import sys
import os
import jieba
from gensim import models
import re

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass


SQL = 'SELECT info_id,content FROM ods_hl_info_ext limit 1' 
print SQL
conn = connect(host='10.129.64.165', port=10000,database='jkgj_log',auth_mechanism='GSSAPI',kerberos_service_name='hive')
cursor = conn.cursor()

cursor.execute(SQL)
lines = cursor.fetchall()
#print lines

regex=u"[\u4e00-\u9fa5]+"
p = re.compile(regex)

#stopkey=[line.strip() for line in open('/home/rd/python/workspace/duankai/dict/stop_word.txt').readlines()]

print("start seperate words...")
sentences = []
for line in lines:
    info_id = line[0]
    info_content = unicode(line[1]).replace("微软雅黑","")
    #print(info_content)    
    result = p.findall(info_content)
    #print("result======",str(result))
    newstr = str(result).replace("\'","").replace(",","").replace(" ","")
    if len(newstr) < 50:
        continue
    #print("newstr:  == ",str(newstr))
    seg_list = jieba.cut(newstr, cut_all = True)
    
    sentence = models.doc2vec.LabeledSentence(words=list(seg_list), tags=[info_id])
    sentences.append(sentence)

print("length: ",len(sentences))    
print("start build model...")
model = models.Doc2Vec(sentences,alpha=.025, min_alpha=.025, min_count=1,workers=8, iter=13, size=500)
#model.build_vocab(sentences)
model.train(sentences)

print("save model")

#for epoch in range(10):
#model.train(sentences)
  #  model.alpha -= 0.002  # decrease the learning rate`
   # model.min_alpha = model.alpha  # fix the learning rate, no decay

model.save("/home/rd/python/workspace/duankai/similarity_model.doc2vec")

print("sucess")
#print(model.docvecs.most_similar(["SENT_0"]))
