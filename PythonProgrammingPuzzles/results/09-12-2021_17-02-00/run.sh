conda activate TransformerCodeComparison_GPU
python src/run_programming_puzzles_experiments.py \
	-n=30 \
	--engine=EleutherAI/gpt-neo-2.7B \
	--filename=397puzzles.json \
	--output_folder=results