# Generating Transformer Results for APPs

The code is largely based from the original [APPs dataset repo](https://github.com/hendrycks/apps) to generate results for the APPs dataset with support for the [gpt2 family](https://huggingface.co/docs/transformers/model_doc/gpt2) of models and OpenAI Codex and GPT-3. The following shows the steps to obtain results. 

### 1. Download the APPs Dataset
To begin download the [**APPS dataset here**](https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz). (~1.3GB)

```bash
wget https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz
tar -xzvf APPs.tar.gz
```
move this to APPS/datasets where the train problems are in APPS/datasets/train and the test problems are in APPS/datasets/test

The folder structure for the dataset is as follows
------------


    ├── train   <- Train has the same layout as the test folder
    │  
    ├── test
    │   ├── 0000           <- Each folder is the problem number
    │   ├── 0001  
    │       ├── question.txt            <- The coding question and used as part of prompt
    │       ├── input_output.json       <- The test cases with inputs and expected outputs (used by test_gpt_codes_solution.py)
    │       ├── metadata.json
    │       ├── solutions.json
    │   ├── ...     
    │   └── 4999           


To include the trained GPT-2 and GPT-Neo models on APPS, get the [saved models here](https://drive.google.com/file/d/1XW1Od9L-5l9zXl1HUCyER5pS9zQTbIvU/view?usp=sharing)


### 2. Get the absolute paths to the APPs problems
```bash
cd APPS
python src/create_dataset_paths_to_problems.py --train_path {path_to_apps_train_folder} --test_path {path_to_apps_test_folder}
```

### 3. Generate Codes for APPs problems with a transformer

The script to do this is `src/generate_gpt_codes.py`. An example command to run this is at `sample_scripts/sample_run_generate_gpt_codes.sh`.

For running OpenAI's [Codex model](https://openai.com/blog/openai-codex/) we need to specify the `OPENAI_API_KEY` envionment variable and set our engine to one of 
* cushman-codex
* davinci-codex

There is also an included cache to save results so we do not have to rerun or recall the API. I've included my cached results in the repo but if you do not want it simply delete the `src/.cache` folder. 

### 4. Test the generated codes against APPS test cases
Once we obtain the generated codes for APPs, we will have a file all_codes*.jsonl file containing the generated code results. 

To evaluate the results, we run `src/test_gpt_codes_solution.py`. It is advised from the original [APPs dataset repo](https://github.com/hendrycks/apps/tree/main/eval) to run this script in a loop as it may fail on some problems due to poorly generated code.

An example command is at `sample_scripts/sample_run_test_gpt_codes_solution.sh`.
This will create a `results_{problem_ind}.jsonl` file for each problem for all the test cases of a given problem specifried in the APPs dataset

### 5. Gather results
To get overall metrics of our results we run `parse_results.py` while specifying the folder contain `results_{problem_ind}.jsonl` files.


