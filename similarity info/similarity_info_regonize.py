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


SQL = 'SELECT info_id,content FROM ods_hl_info_ext' 
print SQL
conn = connect(host='', port=10000,database='jkgj_log',auth_mechanism='GSSAPI',kerberos_service_name='hive')
cursor = conn.cursor()

cursor.execute(SQL)
lines = cursor.fetchall()
#print lines

regex=u"[\u4e00-\u9fa5]+"
p = re.compile(regex)

print("start seperate words...")
sentences = []
for line in lines:
    info_id = line[0]
    info_content = unicode(line[1])
    #print(info_content)    
    result = p.findall(info_content)
    #print("result======",str(result))
    newstr = str(result).replace("\'","").replace(",","").replace(" ","")
    
    #print("newstr:  == ",str(newstr))
    seg_list = jieba.cut(newstr, cut_all = True)
    
    
    sentence = models.doc2vec.LabeledSentence(words=list(seg_list), tags=[info_id])
    sentences.append(sentence)

print("length: ",len(sentences))    
print("start build model...")
model = models.Doc2Vec(sentences,alpha=.025, min_alpha=.025, min_count=1)
#model.build_vocab(sentences)
model.train(sentences)

print("save model")

#for epoch in range(10):
#model.train(sentences)
  #  model.alpha -= 0.002  # decrease the learning rate`
   # model.min_alpha = model.alpha  # fix the learning rate, no decay

model.save("similarity_model.doc2vec")

print("sucess")
#print(model.docvecs.most_similar(["SENT_0"]))
