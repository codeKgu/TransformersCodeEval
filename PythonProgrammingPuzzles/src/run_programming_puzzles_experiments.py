"""
This script runs the codex experiments.
For GPT-3 experiments see run_gpt3_experiments.py in https://github.com/microsoft/PythonProgrammingPuzzles/tree/v0.1
It uses cacheing mechanisms so that if run twice with the same parameters, it will give exactly the same
results and will not query the API again and will not judge the resulting solutions again. Hence, the first
time you run it, it will be slow, but you can subsequently run it again and it will be fast. It will run the
experiment three times, with different seeds to get different results.
"""
import os

import lm_solve
import utils
import numpy as np
import torch
import transformers
import random

SEEDS = 1 # number of times to run it

LOCAL_MODELS = ['gpt2', 'EleutherAI', 'EleutherAI/gpt-neo-2.7B'] + transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST

PREFIX_DOCSTR = '''from typing import List

def f1(s: str):
    return "Hello " + s == "Hello world"

def g1():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"

assert f1(g1())

def f2(s: str):
    return "Hello " + s[::-1] == "Hello world"

def g2():
    """Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"[::-1]

assert f2(g2())

def f3(x: List[int]):
    return len(x) == 2 and sum(x) == 3

def g3():
    """Find a list of two integers whose sum is 3."""
    return [1, 2]

assert f3(g3())

def f4(s: List[str]):
    return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)

def g4():
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""
    return ["a"*(i+2)+"b" for i in range(1000)]

assert f4(g4())

def f5(n: int):
    return str(n * n).startswith("123456789")

def g5():
    """Find an integer whose perfect square begins with 123456789 in its decimal representation."""
    return int(int("123456789" + "0"*9) ** 0.5) + 1

assert f5(g5())

'''  # trailing newlines important


def pass_at_k(k: int, successes: int, attempts: int):
    fail_prob = 1.0
    for i in range(k):
        fail_prob *= (attempts - successes)/attempts # gets right answer of 0 when attempts == successes
        attempts -= 1
    return 1.0 - fail_prob

def load_tokenizer_and_model(args):
    if ('EleutherAI' in args.engine):
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    elif 'gpt' in args.engine: # Should handle GPT-2 and GPT-Neo
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.engine)
    elif args.arch in {'codebert'}:
        tokenizer = transformers.RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    else:
        raise NotImplementedError()
    
    if args.load:
        if 'EleutherAI' in args.engine:
            model = transformers.GPTNeoForCausalLM.from_pretrained(args.load)
        else:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.load)
        print(f"Loaded model from {args.load}")
    else:
        if "EleutherAI" in args.engine:
            model = transformers.GPTNeoForCausalLM.from_pretrained(args.engine)
        else:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.engine)
    return tokenizer, model

def run(args, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)    
        
    params = {
        "filename": args.filename,
        "temp":args.temperature,
        "timeout": args.timeout,
        "n": args.n,
        "cache_only": args.cache_only,
        "engine": args.engine,
        "notes": args.notes
    }
    
    if args.engine in LOCAL_MODELS:
        params['tokenizer'], params['model'] = load_tokenizer_and_model(args)

    sols = [
        lm_solve.prompt_experiment(**params, experiment="long", prefix=PREFIX_DOCSTR, 
                                   remove_docstring=False, seed=seed),
    ]
    
    num_puzzles = len(sols[0]["sat_sols"])
    assert all(len(s["sat_sols"]) == num_puzzles for s in sols)

    n = params["n"]
    ks = [1]
    while ks[-1] < n:
        ks += [ks[-1] * i for i in [10]] # for i in [2, 5, 10]]
    ks = [k for k in ks if k <= n]
    if ks[-1] != n:
        ks.append(n)
    for s in sols:
        s["pass@k"] = [(k, np.mean([pass_at_k(k, s_s["n_sols"], n) for s_s in s["sat_sols"]]))
                       for k in ks]


    print(f"run={seed} ALL DONE!\n\n")
    print(f"run={seed} RESULTS " + "=" * 50)
    print()

    for s in sols:
        print(s["experiment"], "prefix:", s["prefix"].replace("\n", "\\n")[:250])
        print("   ", s["tot_solved"], "solved, pass@k", " ".join(f'{k} {p:.5f}' for k, p in s["pass@k"]))

    print(f"Pass at k [(k, {', '.join(s['experiment'] for s in sols)}) ...]")
    print(list(zip([k for k, _p in sols[0]["pass@k"]], *[[p for _k, p in s["pass@k"]] for s in sols])))

    return sols

def main(args):
    res = [s for seed in range(SEEDS) for s in run(args, seed)]
    output_filename = os.path.join(args.output_folder, f'results_{os.path.basename(args.filename).split(".")[0]}_{args.engine.replace("/", "_")}_{args.n}.json')
    if output_filename:
        full_filename = output_filename.replace(".json", "_full.json.gz")
        utils.save_json(res, full_filename)
        for s in res:
            if "sat_sols" in s:
                for t in s["sat_sols"] :
                    if "sol_counts" in t:
                        if t["sol_counts"]:
                            t["shortest_sol"] = min([s for s, c in t["sol_counts"]], key=len)
                            t["longest_sol"] = max([s for s, c in t["sol_counts"]], key=len)
                            t["common_sol"] = max(t["sol_counts"], key=lambda z: z[1])[0]
                        del t["sol_counts"]
        utils.save_json(res, output_filename)
        print(f"saved results to {output_filename} and {full_filename}")



if __name__ == "__main__":
    import argparse
    import sys
    from datetime import datetime


    
    parser = argparse.ArgumentParser(description="Run OpenAI GPT or Disk GPT")

    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--timeout", type=float, default=1.0, help="seconds to judge for OpenAI API")
    parser.add_argument("-n", type=int, default=32, help="number of attempts per puzzle. For OpenAI API could set large. For local disk not too large")
    parser.add_argument("--filename", type=str, help="path to puzzles json ex. 30puzzles.json")
    parser.add_argument("--engine", type=str, default='cushman-codex', 
                        help="For OpenAI API, the engine used. If engine='gpt2' or engine='EleutherAI' it is the HuggingFace Transformers model we use locally",
                        choices=["cushman-codex", "davinci-codex", "davinci", "gpt2", "EleutherAI", "EleutherAI/gpt-neo-2.7B"] + transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST)
    
    parser.add_argument("-c", "--cache_only", action="store_true", help="change this to True if you want to run a 2nd time without risking hitting API")
    parser.add_argument("--output_folder", type=str, default='/projects/bdata/kenqgu/Courses/CSE599_EMFML/PythonProgrammingPuzzles/results')
    parser.add_argument("--load", type=str, help="path to load saved model in disk")
    parser.add_argument("--notes", type=str, help="any notes we want to add to differentiate between caches")
    args = parser.parse_args()
    
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    args.output_folder = os.path.join(args.output_folder, f'{dt_string}')
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    with open(os.path.join(args.output_folder, f"run.sh"), 'w') as f:
        f.write(f'conda activate TransformerCodeComparison_GPU\n')
        command_str = " \\\n\t".join(sys.argv)
        f.write(f'python {command_str}')

    main(args)

