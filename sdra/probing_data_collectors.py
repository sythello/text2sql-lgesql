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

import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize.treebank import TreebankWordDetokenizer

import re
from copy import deepcopy
from typing import List, Dict

# from language.xsp.data_preprocessing import spider_preprocessing, wikisql_preprocessing, michigan_preprocessing

from sdr_analysis.helpers import general_helpers
from sdr_analysis.helpers.general_helpers import SDRASampleError, _wikisql_db_id_to_table_name, EditDistance
from sdr_analysis.helpers.base_graph_data_collector import BaseGraphDataCollector, BaseGraphDataCollector_spider, BaseGraphDataCollector_wikisql
# from sdr_analysis.helpers.link_prediction_collector import LinkPredictionDataCollector

from argparse import Namespace
from utils.example import Example
from utils.batch import Batch
from model.model_utils import Registrable
from model.model_constructor import Text2SQL


class BaseGraphDataCollector_lgesql(BaseGraphDataCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_args['need_loading_schemas'] = True

    def load_model(self, main_args):
        model_path = main_args.model_path

        params = json.load(open(os.path.join(model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
        params.lazy_load = True
        Example.configuration(plm=params.plm, method=params.model)

        sql_trans, evaluator = Example.trans, Example.evaluator

        model = Registrable.by_name('text2sql')(params, sql_trans).to(self.device_name)

        check_point = torch.load(open(os.path.join(model_path, 'model.bin'), 'rb'), map_location=self.device_name)
        model.load_state_dict(check_point['model'])
        print("Loaded saved model from path: %s" % (model_path))

        # model.to(self.device_name)
        model.eval()
        self.model = model

        return model

    @overrides
    def load_dataset_with_graph(self, dataset_path) -> List:
        """ Load the lgesql-preprocessed dataset with rat_sql_graph """
        with open(dataset_path, 'rb') as f:
            orig_dataset = pickle.load(f)
        # import pdb; pdb.set_trace()     # TODO: what happened here?
        for d in orig_dataset:
            d['rat_sql_graph']['relations'] = json.loads(d['rat_sql_graph']['relations'])
        return orig_dataset

    @overrides
    def get_schemas_dict(self, orig_tables_path, db_path) -> Dict:
        """ 
        Args:
            orig_tables_path: path to spider-style lgesql-preprocessed file (xxx.bin; dataset-specific subclasses need to re-implement get_paths_dict()! )
            db_path: path to the databases (not needed for now)
        """
        with open(orig_tables_path, 'rb') as f:
            db_schemas_dict = pickle.load(f)
        print(f'Loaded preprocessed tables from {orig_tables_path}')
        self.db_schemas_dict = db_schemas_dict
        return db_schemas_dict

    def general_fmt_dict_to_schema(self, general_fmt_dict):
        print('* WARNING: BaseGraphDataCollector_lgesql.general_fmt_dict_to_schema(): Shouldn\'t be called!')
        return None

    def get_node_encodings(self, samples, poolinc_func=None):
        """
        Args:
            samples (List[Dict]): each sample must have 'question' for user input and 'rat_sql_graph' for graph info
            pooling_func (Callable): unused in lgesql
        """

        if isinstance(samples, dict):
            # back compatible: passing a single dict
            samples = [samples]
        
        examples = []
        for d in samples:
            examples.append(Example(d, self.db_schemas_dict[d['db_id']]))

        batch = Batch.from_example_list(examples, self.device_name, train=False)
        # encoder_outputs: (bsize, max_q_len + max_t_len + max_c_len, rnn_hidden_size)
        encoder_outputs, batck_mask = self.model.encoder(batch)
        if self.debug:
            print('encoder_output_hidden_states:', encoder_outputs.shape)

        q_len, t_len, c_len = batch.max_question_len, batch.max_table_len, batch.max_column_len
        assert encoder_outputs.size(1) == q_len + t_len + c_len, (encoder_outputs.size(), q_len, t_len, c_len)
        q_encodings, t_encodings, c_encodings = torch.split(encoder_outputs, [q_len, t_len, c_len], dim=1)
        q_encodings = q_encodings.detach().cpu().numpy()
        t_encodings = t_encodings.detach().cpu().numpy()
        c_encodings = c_encodings.detach().cpu().numpy()

        # import pdb; pdb.set_trace()

        batch_node_encodings = []
        valid_in_batch_ids = []
        for in_batch_idx, example in enumerate(batch.examples):
            ex_q_len = batch.question_lens[in_batch_idx]
            ex_t_len = batch.table_lens[in_batch_idx]
            ex_c_len = batch.column_lens[in_batch_idx]

            lgesql_q_enc_list = [q_encodings[in_batch_idx, i] for i in range(ex_q_len)]
            # LGESQL and RATSQL tokens may mismatch; do an alignment with simple edit distance
            lgesql_q_toks = example.ex['processed_question_toks']
            # ratsql_q_toks = [n for n in example.ex['rat_sql_graph']['nodes'] if not n.startswith("<")]    # Wrong; there are "<" tokens!
            ratsql_q_toks = [n for n in example.ex['rat_sql_graph']['nodes'] if
                (not n.startswith("<C>")) and (not n.startswith("<T>"))]
            _, match_pairs_idx = EditDistance(lgesql_q_toks, ratsql_q_toks, return_pairs_idx=True)
            r2l_idx_dict = dict()
            # For each target token, find the first match source token and use its encoding (here "match" can be a mismatch but skipped at the same step)
            # May have multiple target tokens mapping to the same source token, which is ok, just reuse that source encoding multiple times
            for l_id, r_id in match_pairs_idx:
                if r_id not in r2l_idx_dict:
                    r2l_idx_dict[r_id] = l_id
            ratsql_q_enc_list = []
            for r_id in range(len(ratsql_q_toks)):
                l_id = r2l_idx_dict[r_id]
                ratsql_q_enc_list.append(lgesql_q_enc_list[l_id])

            c_enc_list = [c_encodings[in_batch_idx, i] for i in range(ex_c_len)]
            t_enc_list = [t_encodings[in_batch_idx, i] for i in range(ex_t_len)]
            
            nodes_enc_list = ratsql_q_enc_list + c_enc_list + t_enc_list
            batch_node_encodings.append(nodes_enc_list)
            valid_in_batch_ids.append(in_batch_idx)

        return batch_node_encodings, valid_in_batch_ids
    


class BaseGraphDataCollector_lgesql_spider(BaseGraphDataCollector_lgesql, BaseGraphDataCollector_spider):
    @overrides
    def get_paths_dict(self) -> Dict:
        """ Get the path of orig_dataset_path, orig_tables_path, db_path """
        orig_ds_list = self.get_orig_ds_list()
        paths_dict = {
            orig_ds : {
                'orig_dataset_path': os.path.join(self.orig_dataset_dir, f'{orig_ds}.json'),
                'orig_tables_path': os.path.join(self.graph_dataset_dir, 'tables.bin'),     # For lgesql, the tables.bin file is in graph_dataset_dir (SDRA data dir)
                'db_path': os.path.join(self.orig_dataset_dir, 'database'),                 # For spider, db_path is database root dir
                'graph_dataset_path': os.path.join(self.graph_dataset_dir, f'{orig_ds}+ratsql_graph.lgesql.bin'),
            }
            for orig_ds in orig_ds_list
        }
        return paths_dict

class BaseGraphDataCollector_lgesql_wikisql(BaseGraphDataCollector_lgesql, BaseGraphDataCollector_wikisql):
    def get_paths_dict(self) -> Dict:
        """ Get the path of orig_dataset_path, orig_tables_path, db_path """
        orig_ds_list = self.get_orig_ds_list()
        paths_dict = {
            orig_ds : {
                'orig_dataset_path': os.path.join(self.orig_dataset_dir, f'{orig_ds}.jsonl'),
                'orig_tables_path': os.path.join(self.graph_dataset_dir, f'{orig_ds}.spider-fmt-tables.bin'),   # For lgesql, the tables.bin file is in graph_dataset_dir (SDRA data dir)
                'db_path': os.path.join(self.orig_dataset_dir, f'{orig_ds}.db'),    # For wikisql, db_path is the db file path
                'graph_dataset_path': os.path.join(self.graph_dataset_dir, f'{orig_ds}+ratsql_graph.lgesql.bin'),
            }
            for orig_ds in orig_ds_list
        }
        return paths_dict
