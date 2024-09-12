#!/bin/bash
. /home/peb/ws/hadoop-env/cm-hadoop-2-cdh5.14.4-ARCH/setup-env.sh

hadoop jar $HADOOP_INSTALL_DIR/hadoop-mapreduce/hadoop-streaming.jar \
  -D mapreduce.job.maps=3 \
  -D mapreduce.job.reduces=1 \
  -mapper "jsonl2img_embedding_mapper.py ViT-L-14 laion2b_s32b_b82k" \
  -reducer "jsonl2img_embedding_reducer.py" \
  -input gifcities/jsonl_sample \
  -output gifcities/output \
  -file jsonl2img_embedding_mapper.py \
  -file jsonl2img_embedding_reducer.py
