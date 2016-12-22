#!usr/bin/python
# -*- coding: utf-8 -*

import codecs
from impala.dbapi import connect
from impala.util import as_pandas
from gensim import models

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass


SQL = 'SELECT info_id FROM ods_hl_info_ext'
print SQL
conn = connect(host='', port=10000,database='jkgj_log',auth_mechanism='GSSAPI',kerberos_service_name='hive')
cursor = conn.cursor()

cursor.execute(SQL)
info_ids = cursor.fetchall()
lines = []

model = models.Doc2Vec.load("/home/rd/python/workspace/duankai/similarity_model.doc2vec")
for info_id in info_ids:
    line = ""
    infoid = info_id[0]
    line = infoid + "\t"
    similar_result = model.docvecs.most_similar(infoid, topn=10)
    similar_ids = []
    for (s_infoid, similarity) in similar_result:
        if similarity > 0.75:
            similar_ids.append(s_infoid)
        else:
            break

    if len(similar_ids) > 0:
        line = line + str(similar_ids) + "\n"
        line = line.replace("\'","").replace("[","").replace("]","")
        lines.append(line)


file_write = codecs.open("/home/rd/python/workspace/duankai/similarity_ids.txt",'w+','utf-8')
file_write.writelines(lines)
print("Write file sucessfully!")
file_write.close()
