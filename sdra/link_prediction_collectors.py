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

from sdr_analysis.helpers import general_helpers
from sdr_analysis.helpers.general_helpers import SDRASampleError
from sdr_analysis.helpers.base_graph_data_collector import BaseGraphDataCollector, BaseGraphDataCollector_spider, BaseGraphDataCollector_wikisql
from sdr_analysis.helpers.link_prediction_collector import LinkPredictionDataCollector, collect_link_prediction_samples
from sdra.probing_data_collectors import BaseGraphDataCollector_lgesql, BaseGraphDataCollector_lgesql_spider, BaseGraphDataCollector_lgesql_wikisql


class LinkPredictionDataCollector_lgesql(LinkPredictionDataCollector, BaseGraphDataCollector_lgesql):
    pass


class LinkPredictionDataCollector_lgesql_spider(LinkPredictionDataCollector_lgesql, BaseGraphDataCollector_lgesql_spider):
    pass


class LinkPredictionDataCollector_lgesql_wikisql(LinkPredictionDataCollector_lgesql, BaseGraphDataCollector_lgesql_wikisql):
    pass

