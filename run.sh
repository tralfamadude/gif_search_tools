#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate gifcities_exp
echo "ENV is $CONDA_DEFAULT_ENV"
BASE=full_jsonl

for j in full_jsonl/*gz ; do  
	INPUT="full_jsonl/$(basename $j .gz).jsonl"
	OUTPUT="full_jsonl/$(basename $j .gz).embedding.jsonl"
	[ -r $OUTPUT ]  && echo "SKIP $INPUT: already processed" &&  continue
	zcat $j > $INPUT   # unpack into input for next step
	echo "-----------------${INPUT} -> ${OUTPUT}---------$(date '+%Y%m%dt%H%M')---"
	python json2img_embedding_standalone.py --output_file $OUTPUT $INPUT --model_name ViT-L-14 --pretrained laion2b_s32b_b82k  --k 3 --neighborhood_threshold 0.05  
done
