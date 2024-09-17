#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate gifcities_exp
echo "ENV is $CONDA_DEFAULT_ENV"
sleep 10
BASE=full_jsonl

for j in ${BASE}/*gz ; do  
	INPUT="${BASE}/$(basename $j .gz).jsonl"
	OUTPUT="${BASE}/$(basename $j .gz).embedding.jsonl"
	[ -r $OUTPUT ]  && echo "SKIP $INPUT: already processed" &&  continue
	zcat $j > $INPUT   # unpack into input for next step
	echo "-----------------${INPUT} -> ${OUTPUT}---------$(date '+%Y%m%dt%H%M')---"
	python json2img_embedding_standalone.py --output_file $OUTPUT $INPUT --model_name ViT-L-14 --pretrained laion2b_s32b_b82k  --k 3 --neighborhood_threshold 0.05  
done
