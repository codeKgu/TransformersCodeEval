"""
Run a tranined model to generate Python code.
"""

import argparse
from datetime import datetime
import io
import json
import random
import os

import jsonlines
import pprint
import transformers
from tqdm import tqdm

from utils.reindent import run as run_reindent
from utils.gpt_lib import LOCAL_MODELS, CACHES, BATCH_SIZES, load_tokenizer_and_model, call_gpt_for_results


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()

def generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None):
    _input = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data #+ "\n"
        _input += data
    else:
        #_input += "\n\n"
        pass

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"#\n"
    else:
        _input += "\nUse Call-Based format"#\n"
    
    _input += "\nANSWER:\n"

    if args.peeking > 0.0:
        # Need to do some peeking. 

        # Read one example solution
        with open(solutions_path, 'r') as f:
            sols = json.load(f)

        # Choose the shortest solution for the model to use.
        # This is so we can conserve tokens (1024 max)
        # sample_sol = min(sols, key=len)

        # # Add args.peeking% of that solution to the prompt
        # sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
        # num_to_keep = int(len(sample_sol_token_ids) * args.peeking)
        # sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
        # _input += tokenizer.decode(sample_sol_token_ids)

        # Alternatively take a random solution
        sample_sol = random.choice(sols)
        rand_sol = reindent_code(sample_sol)
        rand_sol = tokenizer.encode(rand_sol, verbose=False)
        tokens_taken = int(args.peek_frac * len(rand_sol))
        rand_sol = rand_sol[:tokens_taken]
        _input += tokenizer.decode(rand_sol)
    else:
        sample_sol = None

    return _input, sample_sol


def get_code_save_path(args):
    
    now = datetime.now()
    print("now =", now)
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    if not args.end:
        codes_loc = os.path.join(args.save, f"all_codes_{dt_string}.jsonl")
    else:
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes_{dt_string}.jsonl")

    return codes_loc
    
def get_problems(args):
    
    with open(args.test_loc, "r") as f:
        problems = json.load(f)
    problems = sorted(problems) # Pin some ordering
    
    # Only do the problems that are specified.
    if args.index:
        problems = [problems[args.index]]
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = problems[start:end]
    return problems
    
def main(args):

    global BATCH_SIZES
    global CACHES
    cur_cache = CACHES[args.engine]
    max_batch = BATCH_SIZES[args.engine]
    
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    
    codes_loc = get_code_save_path(args)
    problems = get_problems(args)

    # Tokenizer
    tokenizer, model = None, None
    if args.engine in LOCAL_MODELS:
        print(f'Loading model {args.engine} to be locally ran')
        tokenizer, model = load_tokenizer_and_model(args)
        model.cuda()

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        prob_path = os.path.join(args.root, problem)
        if args.debug:
            print(f"problem path = {prob_path}")

        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")
        if not os.path.exists(starter_path):
            starter_path = None
        if not os.path.exists(test_case_path) or not os.path.exists(prompt_path):
            continue

        # Read the question in
        prompt_text, sample_sol = generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path)
        if args.debug:
            print("PROMPT_TEXT:")
            print(prompt_text)
        
        # Feed this into the model.
        key = str(dict(prompt=prompt_text, temp=args.temperature, max_tokens=args.max_tokens))
        cached = cur_cache.get(key)

        if len(cached) >= args.num_samples:
            all_res = cached[:args.num_samples]
        else: 
            assert not args.cache_only, f'Entry not found in cache with prompt "{json.dumps(prompt_text)}"'
            all_res = call_gpt_for_results(model, tokenizer, prompt_text, sample_sol,
                                           args, cached, cur_cache, key, max_batch)
        with jsonlines.open(codes_loc, mode='a') as writer:
            writer.write({index+args.start: all_res})

    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run a tranined model to generate Python code.")
    parser.add_argument("--engine", default="gpt2", choices=transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST + ['EleutherAI', 'EleutherAI/gpt-neo-2.7B'] + ["cushman-codex", "davinci-codex", "davinci"])
    parser.add_argument("-t","--test_loc", default="~/apps/data_split/test.json", type=str)
    parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-l","--load", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("-c", "--cache_only", action="store_true", help="change this to True if you want to run a 2nd time without risking hitting API")
    parser.add_argument("--max_tokens", type=int, default=200, help="used for openai API")
 
    args = parser.parse_args()
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    with open(os.path.join(args.save, f"run.sh"), 'w') as f:
        command_str = " \\\n\t".join(sys.argv)
        f.write(f'python {command_str}')
    main(args)
