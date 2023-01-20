#!/bin/bash

set -e

orig_data_dir='/home/yshao/Projects/language/language/xsp/data/wikisql'
sdra_data_dir='/home/yshao/Projects/SDR-analysis/data/wikisql'

train_data="${sdra_data_dir}/train+ratsql_graph.json"
dev_data="${sdra_data_dir}/dev+ratsql_graph.json"
test_data="${sdra_data_dir}/test+ratsql_graph.json"
train_table_data="${orig_data_dir}/train.spider-fmt-tables.json"
dev_table_data="${orig_data_dir}/dev.spider-fmt-tables.json"
test_table_data="${orig_data_dir}/test.spider-fmt-tables.json"
train_db="${orig_data_dir}/train.db"
dev_db="${orig_data_dir}/dev.db"
test_db="${orig_data_dir}/test.db"

tmp_train_out="${sdra_data_dir}/train+ratsql_graph.bin"
tmp_dev_out="${sdra_data_dir}/dev+ratsql_graph.bin"
tmp_test_out="${sdra_data_dir}/test+ratsql_graph.bin"

train_out="${sdra_data_dir}/train+ratsql_graph.lgesql.bin"
dev_out="${sdra_data_dir}/dev+ratsql_graph.lgesql.bin"
test_out="${sdra_data_dir}/test+ratsql_graph.lgesql.bin"
train_table_out="${sdra_data_dir}/train.spider-fmt-tables.bin"
dev_table_out="${sdra_data_dir}/dev.spider-fmt-tables.bin"
test_table_out="${sdra_data_dir}/test.spider-fmt-tables.bin"

vocab_glove='pretrained_models/glove.42b.300d/vocab_glove.txt'
vocab="${sdra_data_dir}/vocab.txt"

# echo "Start to preprocess the original train (+ratsql_graph) dataset ..."
# # python3 -u preprocess/process_dataset.py --dataset_name wikisql --dataset_path ${train_data} --raw_table_path ${train_table_data} --table_path ${train_table_out} --output_path ${tmp_train_out} --verbose > train.wikisql.log
# python3 -u preprocess/process_dataset.py --dataset_name wikisql --dataset_path ${train_data} --db_dir ${train_db} --table_path ${train_table_out} --output_path ${tmp_train_out} --verbose > train.wikisql2.log

# echo "Start to preprocess the original dev (+ratsql_graph) dataset ..."
# python3 -u preprocess/process_dataset.py --dataset_name wikisql --dataset_path ${dev_data} --db_dir ${dev_db} --raw_table_path ${dev_table_data} --table_path ${dev_table_out} --output_path ${tmp_dev_out} #--verbose > dev.wikisql.log

# echo "Start to preprocess the original test (+ratsql_graph) dataset ..."
# python3 -u preprocess/process_dataset.py --dataset_name wikisql --dataset_path ${test_data} --db_dir ${test_db} --raw_table_path ${test_table_data} --table_path ${test_table_out} --output_path ${tmp_test_out} #--verbose > test.wikisql.log

echo "Start to build word vocab for the dataset ..."
python3 -u preprocess/build_glove_vocab.py --data_paths ${tmp_train_out} --table_path ${train_table_out} --reference_file ${vocab_glove} --mwf 4 --output_path ${vocab}

echo "Start to construct graphs for the dataset ..."
python3 -u preprocess/process_graphs.py --dataset_path ${tmp_train_out} --table_path ${train_table_out} --method 'lgesql' --output_path ${train_out}
python3 -u preprocess/process_graphs.py --dataset_path ${tmp_dev_out} --table_path ${dev_table_out} --method 'lgesql' --output_path ${dev_out}
python3 -u preprocess/process_graphs.py --dataset_path ${tmp_test_out} --table_path ${test_table_out} --method 'lgesql' --output_path ${test_out}
