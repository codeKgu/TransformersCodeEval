# Generating Transformer Results for Python Programming Puzzles

The code is largely based from the original [Python Programming Puzzles repo](https://github.com/microsoft/PythonProgrammingPuzzles) to generate results for the Python Programming Puzzles dataset with support for the [gpt2 family](https://huggingface.co/docs/transformers/model_doc/gpt2) of models and OpenAI Codex and GPT-3. 


To include the trained GPT-2 and GPT-Neo models on APPS, get the [saved models here](https://drive.google.com/file/d/1XW1Od9L-5l9zXl1HUCyER5pS9zQTbIvU/view?usp=sharing)


Running the results is simple in this case. Just run `run_codex_experiments.py`

``` python
export PYTHONHASHSEED="0" # set this for determinism

python src/run_programming_puzzles_experiments.py \
	-n=30 \
	--engine=EleutherAI/gpt-neo-2.7B \
	--filename=397puzzles.json \
	--output_folder=results
```

For OpenAI's Codex, set the OPENAI_API_KEY environment variable to your api key.

``` python
export PYTHONHASHSEED="0" # set this for determinism
export OPENAI_API_KEY="YOUR_API_KEY" 

python src/run_programming_puzzles_experiments.py \
	-n=30 \
	--engine=davinci-codex \
	--filename=397puzzles.json \
	--output_folder=results
```

There is also an included cache to save results so we do not have to rerun or recall the API. I've included my cached results in the repo but if you do not want it simply delete the `src/.cache` folder. 