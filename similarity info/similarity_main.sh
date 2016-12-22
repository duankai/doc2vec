#! /usr/bin


python /home/rd/python/workspace/duankai/calculate_similarity_info.py

hive -e "use jkgj_log; drop table dk_similarity_infos_1215; create table dk_similarity_infos_1215(info_id string, similarity_ids str
ing)ROW FORMAT DELIMITED fields terminated by '\t';"

hive -e "load data local inpath '/home/rd/python/workspace/duankai/similarity_ids.txt' into table jkgj_log.dk_similarity_infos_1215;
"
