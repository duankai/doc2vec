#!usr/bin/python
# -*- coding: utf-8 -*

import codecs
from impala.dbapi import connect
from impala.util import as_pandas
from gensim import models
import sys
import copy

p1 = float(sys.argv[1]) if len(sys.argv[1])>0 else 0.75

print("similarity: " ,p1)

if p1 > 1 or p1 <= 0:
    print("parameters are error, must be in 0 and 1")
    exit(1)

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

#has_calculated = []
similar_lines = []
for info_id in info_ids:
    try: 
        infoid = info_id[0]
        similar_result = model.docvecs.most_similar(infoid, topn=30)
        similar_ids = [infoid]
    
        for (s_infoid, similarity) in similar_result:
            if similarity > p1:
                similar_ids.append(s_infoid)
            else:
                break
        if len(similar_ids) > 1:
            similar_lines.append(set(similar_ids))
    except:
        continue


sorted_similar_lines = sorted(similar_lines, key=lambda x:len(x))

sorted_similar_lines_copy = copy.copy(sorted_similar_lines)
for i in range(0, len(sorted_similar_lines)-1):
    v = sorted_similar_lines[i]
    j = i + 1
    for j in range(j, len(sorted_similar_lines)-1):
        if v <= sorted_similar_lines[j]:
            sorted_similar_lines_copy.remove(v)
            break
        j =j + 1
    i = i + 1


line_num = 0

for value in sorted_similar_lines_copy:
    similar_ids = list(value)
    if len(similar_ids) > 0:
        line_num = line_num + 1
        line = str(line_num) + '\t'
        line = line + str(similar_ids) + "\n"
        line = line.replace("\'","").replace("[","").replace("]","").replace(" ","")
        lines.append(line)


file_write = codecs.open("/home/rd/python/workspace/duankai/similarity_ids.txt",'w+','utf-8')
file_write.writelines(lines)
print("Write file sucessfully!")
file_write.close()
