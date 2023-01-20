import sys
import os
import time
import torch

import json
from copy import deepcopy
from collections import Counter, defaultdict
import importlib
import pickle
import random
from overrides import overrides
import faulthandler
import signal
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize.treebank import TreebankWordDetokenizer

import re
from copy import deepcopy
from typing import List, Dict

from language.xsp.data_preprocessing import spider_preprocessing, wikisql_preprocessing, michigan_preprocessing

from sdr_analysis.helpers.general_helpers import SDRASampleError
from sdra import probing_data_utils as pb_utils
from sdra.link_prediction_collectors import LinkPredictionDataCollector_lgesql_spider, LinkPredictionDataCollector_lgesql_wikisql
from sdra.single_node_reconstruction_collectors import SingleNodeReconstructionDataCollector_lgesql_spider, SingleNodeReconstructionDataCollector_lgesql_wikisql


DATA_COLLECTOR_CLS_DICT = {
    "spider": {
        "link_prediction": LinkPredictionDataCollector_lgesql_spider,
        "single_node_reconstruction": SingleNodeReconstructionDataCollector_lgesql_spider,
    },
    "wikisql": {
        "link_prediction": LinkPredictionDataCollector_lgesql_wikisql,
        "single_node_reconstruction": SingleNodeReconstructionDataCollector_lgesql_wikisql,
    },
}


def main(args):
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1.value)

    # if args.dataset == 'spider':
    #     probe_data_collector_cls = LinkPredictionDataCollector_USKG_spider
    # elif args.dataset == 'wikisql':
    #     probe_data_collector_cls = LinkPredictionDataCollector_USKG_wikisql
    # else:
    #     raise ValueError(args.dataset)
    probe_data_collector_cls = DATA_COLLECTOR_CLS_DICT[args.dataset][args.probe_task]
    
    probe_data_collector = probe_data_collector_cls(
        orig_dataset_dir=args.orig_dataset_dir,
        graph_dataset_dir=args.graph_dataset_dir,
        probing_data_in_dir=args.probing_data_in_dir,
        probing_data_out_dir=args.probing_data_out_dir,
        max_label_occ=args.max_label_occ,
        ds_size=args.ds_size,
        enc_batch_size=args.enc_batch_size,
        device_name='cuda:0' if args.gpu else 'cpu',
    )

    # probe_data_collector.orig_ds_list = ['train_others']
    # probe_data_collector.prob_ds_list = ['test']
    # probe_data_collector._start_idx = 12
    # probe_data_collector._end_idx = 30

    probe_data_collector.load_model(args)

    probe_data_collector.collect_all_probing_datasets()



if __name__ == '__main__':
    parser = ArgumentParser()

    # uskg-specific args
    parser.add_argument('-model', '--model_path', type=str, required=False, default='saved_models/electra-msde-75.1',
        help="The model to use. (Provided: electra-msde-75.1 / bert-msde-74.1")
    # parser.add_argument('-cfg', '--uskg_config', type=str, required=False, default='Salesforce/T5_large_prefix_spider_with_cell_value.cfg',
    #     help="The USKG config file (Salesforce/xxx) to use.")

    # general args
    parser.add_argument('-ds', '--dataset', type=str, required=True,
        help="Which dataset; now support 'spider' and 'wikisql'.")
    parser.add_argument('-probe_task', '--probe_task', type=str, required=True,
        help="Which probing task.")
    # parser.add_argument('-tables_path', '--tables_path', type=str, required=True,
    #     help="Input spider tables file (tables.json)")
    # parser.add_argument('-db_path', '--db_path', type=str, required=True,
    #     help="Path to databases. spider: db dir; wikisql: db file")
    parser.add_argument('-orig_dataset_dir', '--orig_dataset_dir', type=str, required=True,
        help="Dir with original input dataset files (e.g. .../xsp/data/{dataset})")
    parser.add_argument('-graph_dataset_dir', '--graph_dataset_dir', type=str, required=True,
        help="Dir with graph-preprocessed input dataset files (e.g. .../SDR-analysis/data/{dataset})")
    parser.add_argument('-pb_in_dir', '--probing_data_in_dir', type=str, required=False,
        help="The directory with input probing data files (to load pos file from)")
    parser.add_argument('-pb_out_dir', '--probing_data_out_dir', type=str, required=True,
        help="The directory to have output probing data files (for uskg)")
    parser.add_argument('-enc_bsz', '--enc_batch_size', type=int, required=False, default=1,
        help="Batch size when computing the encodings.")
    parser.add_argument('-ds_size', '--ds_size', type=int, required=False, default=None,
        help="Only used when no 'pb_in_dir' given. Use X samples from original dataset to collect probing samples.")
    parser.add_argument('-max_label_occ', '--max_label_occ', type=int, required=False, default=None,
        help="Only used when no 'pb_in_dir' given. For each spider sample, include at most X probing samples per relation type.")

    parser.add_argument('-gpu', '--gpu', action='store_true', default=False,
        help="If set, use GPU.")


    args = parser.parse_args()

    main(args)





