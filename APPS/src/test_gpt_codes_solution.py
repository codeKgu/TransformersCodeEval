"""
Run solutions from one problem.
"""

import io
import json
import logging
import math
import numpy as np
import os
import pprint
import sys
from utils import testing_util as test_util
import time

# for timing debugging
import jsonlines
from datetime import datetime, date
from tqdm import tqdm

from typing import List

def convert_res_to_true_false(results):
    ret = []
    for res in results:
        ret.append([False if test_case == -1 or test_case == -2 else test_case for test_case in res])
    return ret

def print_results(results, args):
    res = []
    per_prob_res = []
    all_correct = []
    for index in results:
        cur_res = results[index]
        res.extend(cur_res)
        per_prob_res.append(np.mean(convert_res_to_true_false(cur_res)))
        all_correct.append(np.all(convert_res_to_true_false(cur_res)))
    tmp_results = res
    compile_errors_bool = [test_case ==-2 for result in tmp_results for test_case in result]
    compile_errors = sum(compile_errors_bool)
    runtime_errors = sum([test_case ==-1 for result in tmp_results for test_case in result])
    failures = sum([test_case ==False for result in tmp_results for test_case in result])
    successes = sum([test_case ==True for result in tmp_results for test_case in result])
    total_testcases = len(compile_errors_bool)
    if args.debug:
        print(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases}")
        print(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}")
        print(f"number of test cases run = {total_testcases}")

    print(f"Test Case Average (average accuracy over problems) = {np.mean(per_prob_res)}")
    print(f"Strict Accuracy (all test cases passed / total problems) = {np.mean(all_correct)}")

def load_gpt_codes(args, problems):
    codes_loc = args.save
    if not args.jsonl:
        with open(codes_loc, "r") as f: 
            gpt_codes = json.load(f)
    else:
        codes_loc = args.save
        with open(codes_loc, 'r') as json_file:
            gpt_codes = list(map(json.loads, json_file))    
            gpt_codes = dict((key, val) for k in gpt_codes for key, val in k.items())
    
    if os.path.exists(codes_loc):
        results_loc = os.path.join(os.path.dirname(args.save), f"all_results.json") 
    else:
        results_loc = os.path.join(os.path.dirname(args.save), f"{args.start}-{args.end}_results.json") 
    print(codes_loc, results_loc)
    
    if args.index >= 0:
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
    if len(gpt_codes) < len(problems):
        subset_problems = [problems[int(i)] for i in gpt_codes.keys()]
        problems = subset_problems
    return problems, gpt_codes, results_loc

def eval_and_save_problems(args):
    with open(args.test_loc, "r") as f:
        problems = sorted(json.load(f))

    problems, gpt_codes, results_loc = load_gpt_codes(args, problems)

    print(len(problems))
    # gpt_codes = {}
    gpt_bleu = {}
    gpt_codebleu = {}
    results = {}

    
    if args.index >= 0:
        new_folder = os.path.join(os.path.dirname(args.save), 'test_results')
        if not os.path.exists(new_folder):
            os.makedirs(new_folder, exist_ok=True)
        results_jsonl_loc = os.path.join(new_folder, f"results_{args.index}.jsonl") 
    else:
        now = datetime.now()
        print("now =", now)
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        results_jsonl_loc = os.path.join(os.path.dirname(args.save), f"all_results_{dt_string}.jsonl") 
    if args.stop_early:
        problems = problems[:args.stop_early]

    # main eval loop
    for index, problem in enumerate(tqdm(problems, desc='problems')):
        try:
            if args.debug:
                print(f"\n\nproblem path = {problem}")
            output_str = gpt_codes[str(index+args.start)] if args.index < 0 else gpt_codes[str(args.index)]
        except:
            print("CANNOT FIND OUTPUT_STR FOR", problem)
            continue
        prob_path = os.path.join(args.root, problem)

        # with open(os.path.join(prob_path, "solutions.json"), "r") as f:
        #     sols = json.load(f)
        
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        if type(output_str) is str:
            output_str = [output_str]
        res = []
        for o_idx, o in enumerate(tqdm(output_str, desc='result in problem')):

            curr_res = [-2]
            if 'if __name__ == "__main__":' in o:
                before_str, after_str = o.split('if __name__ == "__main__":')
                after_lines = after_str.split("\n")
                after_str = '\n'.join([line.lstrip() for line in after_lines])
                o = before_str + after_str
            print(o)
            print(f"\nTesting solution {o_idx}")
            print("now =", datetime.now())


            try:
                curr_res = test_util.run_test(prob_path=prob_path, test=o, debug=args.debug)
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                       e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                # if not np.all(curr_res):
                #     print(f"Results were not all True: {curr_res}")
            except Exception as e:
                print(f"test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)
            # TODO KEN the cur_res is the amount of test cases passed for a given problem (from input_output.json) of the problem
            # TODO KEN figure out how to summarize this as an evaluation metric for us to use
        if args.debug:
            print(f"\nHow to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
            #print(f"results = {res}")
        with jsonlines.open(results_jsonl_loc, mode='a') as writer:
            writer.write({index+args.start+args.index: res})
            
        results[index+args.start+args.index] = res
        
        with open(results_loc, "w") as f:
            try:
                f.write(json.dumps(results))
            except Exception as e:
                import pdb; pdb.set_trace()
                print("didn't save problem due to {e}")

    return results


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if args.print_results:
        results = {}
        codes_loc = os.path.join(args.save, f"all_codes.json")
        if os.path.exists(codes_loc):
            results_loc = os.path.join(args.save, f"all_results.json") 
        else:
            results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.json") 
        with open(results_loc, "r") as f: 
            results = json.load(f)
    else:
        results = eval_and_save_problems(args)

    print_results(results, args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("-t","--test_loc", default="../data_split/test.json", type=str, help="path to the json containing problem paths to be evaluated.")
    parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int, help="If you want to evaluate a subset of problems specify start and ending index. File with start and ending prefix must exist typically used with batch evaluation.")
    parser.add_argument("-i", "--index", default=-1, type=int)
    parser.add_argument("-p", "--print_results", action="store_true", help="If you have already evaluated the results and only want to print them.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results", help="Where the evaluated data is loaded from and results saved to.")
    parser.add_argument("--stop-early", default=None, type=int)
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()

    main(args)
