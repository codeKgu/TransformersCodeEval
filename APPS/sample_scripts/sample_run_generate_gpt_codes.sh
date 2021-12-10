python /projects/bdata/kenqgu/Courses/CSE599_EMFML/TransformersCodeEval/APPS/src/generate_gpt_codes.py \
	-t=/projects/bdata/kenqgu/Courses/CSE599_EMFML/apps/train/test.json \
	--root=/projects/bdata/kenqgu/datasets/apps_dataset/test \
	--temperature=0.8 \
	--num_samples=1 \
	--save=/projects/bdata/kenqgu/Courses/CSE599_EMFML/TransformersCodeEval/APPS/eval/results/gpt2_1_sample \
	--engine=gpt2 \
	--load=/projects/bdata/kenqgu/Courses/CSE599_EMFML/models/1.5B