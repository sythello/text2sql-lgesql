#!/bin/bash

set -e

orig_data_dir='data'
data_dir='/home/yshao/Projects/SDR-analysis/data/spider'

train_data="${data_dir}/train+ratsql_graph.json"
train_others_data="${data_dir}/train_others+ratsql_graph.json"
dev_data="${data_dir}/dev+ratsql_graph.json"
table_data="${orig_data_dir}/tables.json"       # tables.json only in original spider

tmp_train_out="${data_dir}/train+ratsql_graph.bin"
tmp_train_others_out="${data_dir}/train_others+ratsql_graph.bin"
tmp_dev_out="${data_dir}/dev+ratsql_graph.bin"

train_out="${data_dir}/train+ratsql_graph.lgesql.bin"
train_others_out="${data_dir}/train_others+ratsql_graph.lgesql.bin"
dev_out="${data_dir}/dev+ratsql_graph.lgesql.bin"
table_out="${data_dir}/tables.bin"

vocab_glove='pretrained_models/glove.42b.300d/vocab_glove.txt'
vocab="${data_dir}/vocab.txt"

echo "Start to preprocess the original train (+ratsql_graph) dataset ..."
# python3 -u preprocess/process_dataset.py --dataset_path ${train_data} --raw_table_path ${table_data} --table_path ${table_out} --output_path ${tmp_train_out} --verbose > train.spider.log
python3 -u preprocess/process_dataset.py --dataset_path ${train_data} --table_path ${table_out} --output_path ${tmp_train_out} --verbose > train.spider2.log

# echo "Start to preprocess the original train_others (+ratsql_graph) dataset ..."
python3 -u preprocess/process_dataset.py --dataset_path ${train_others_data} --table_path ${table_out} --output_path ${tmp_train_others_out} #--verbose > train_others.log

# echo "Start to preprocess the original dev (+ratsql_graph) dataset ..."
python3 -u preprocess/process_dataset.py --dataset_path ${dev_data} --table_path ${table_out} --output_path ${tmp_dev_out} #--verbose > dev.log

echo "Start to build word vocab for the dataset ..."
python3 -u preprocess/build_glove_vocab.py --data_paths ${tmp_train_out} --table_path ${table_out} --reference_file ${vocab_glove} --mwf 4 --output_path ${vocab}

echo "Start to construct graphs for the dataset ..."
python3 -u preprocess/process_graphs.py --dataset_path ${tmp_train_out} --table_path ${table_out} --method 'lgesql' --output_path ${train_out}
python3 -u preprocess/process_graphs.py --dataset_path ${tmp_train_others_out} --table_path ${table_out} --method 'lgesql' --output_path ${train_others_out}
python3 -u preprocess/process_graphs.py --dataset_path ${tmp_dev_out} --table_path ${table_out} --method 'lgesql' --output_path ${dev_out}
