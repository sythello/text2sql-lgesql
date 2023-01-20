set -ex

PROJ_DIR='/home/yshao/Projects'
DATASET='wikisql'
# DATASET='spider'
PROBE_TASK='link_prediction'
# PROBE_TASK='single_node_reconstruction'


python -m sdra.probing_data_collect \
-ds ${DATASET} \
-probe_task ${PROBE_TASK} \
-orig_dataset_dir ${PROJ_DIR}/language/language/xsp/data/${DATASET} \
-graph_dataset_dir ${PROJ_DIR}/SDR-analysis/data/${DATASET} \
-pb_out_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/lgesql \
-enc_bsz 2  --gpu \
-pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/uskg



# -pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/uskg \
# -max_label_occ 5
# -ds_size 500
# --gpu
