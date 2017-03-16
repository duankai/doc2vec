#! /usr/bin

#training model doc2vec
#python /home/rd/python/workspace/duankai/similarity_info_regonize.py

#computing similarity info
echo $1
python /home/rd/python/workspace/duankai/calculate_similarity_info.py $1

#create table
hive -e "use jkgj_log; drop table dk_similarity_infos_1215;create table dk_similarity_infos_1215(info_id string, similarity_ids string)ROW FORMAT DELIMITED fields terminated by '\t';"

#load new data
hive -e "load data local inpath '/home/rd/python/workspace/duankai/similarity_ids.txt' into table jkgj_log.dk_similarity_infos_1215;
"
