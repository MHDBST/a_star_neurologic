#!/usr/bin/env bash

export PYTHONPATH=/home/mbastan/context_home/structuralDecoding/a_star_neurologic

DATA_DIR='../dataset/machine_translation'
# DATA_PREFIX='iate.414'
# MODEL_NAME='Helsinki-NLP/opus-mt-en-de'
# INPUT_PATH=${DATA_DIR}/newstest2017-iate/${DATA_PREFIX}.terminology.tsv.en
# CONSTRAINT_FILE= ${DATA_DIR}/constraint/${DATA_PREFIX}.json
# Average score: -1.6023320763181
# Average sum logprob: -13.261617175910784
# Average PPL: 0.4077288284724608

MODEL_NAME='Helsinki-NLP/opus-mt-de-en'
INPUT_PATH=${DATA_DIR}/newstest2019-deen-in.txt
CONSTRAINT_FILE=${DATA_DIR}/newstest2019_deen.constraint.json


DEVICES=$1
OUTPUT_FILE=$2

# neurologic with greedy look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --input_path ${INPUT_PATH} --output_file ${OUTPUT_FILE} \
  --constraint_file ${CONSTRAINT_FILE} \
  --batch_size 2 --beam_size 5 --max_tgt_length 156 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 200 --sat_tolerance 2 --beta 0.25 \
  --look_ahead_step 35  --alpha 0.05 --look_ahead_width 1 #--fusion_t 1.0

# neurologic with sampling look-ahead
# CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
#   --input_path ${DATA_DIR}/newstest2017-iate/${DATA_PREFIX}.terminology.tsv.en --output_file ${OUTPUT_FILE} \
#   --constraint_file ${DATA_DIR}/constraint/${DATA_PREFIX}.json \
#   --batch_size 64 --beam_size 5 --max_tgt_length 156 --min_tgt_length 3 \
#   --length_penalty 0.6 \
#   --prune_factor 200 --sat_tolerance 2 --beta 0.25 \
#   --look_ahead_step 35  --alpha 0.05  --look_ahead_sample --look_ahead_width 5

# # neurologic with beam look-ahead
# CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
#   --input_path ${DATA_DIR}/newstest2017-iate/${DATA_PREFIX}.terminology.tsv.en --output_file ${OUTPUT_FILE} \
#   --constraint_file ${DATA_DIR}/constraint/${DATA_PREFIX}.json \
#   --batch_size 64 --beam_size 5 --max_tgt_length 156 --min_tgt_length 3 \
#   --length_penalty 0.6 \
#   --prune_factor 200 --sat_tolerance 2 --beta 0.25 \
#   --look_ahead_step 35  --alpha 0.05 --look_ahead_width 2


