import argparse
from collections import defaultdict
import json
from os.path import join, basename
from glob import glob

import jsonlines
import numpy as np
from pprint import pprint 
from tqdm import tqdm


def get_problems_difficulty(apps_test_path):
    difficulty_map = {}
    folders = glob(join(apps_test_path, '**'))
    for folder in tqdm(folders):
        ind = int(basename(folder))
        metadata_path = join(folder, 'metadata.json')
        with open(metadata_path, 'rb') as f:
            metadata = json.load(f)
            difficulty_map[ind] = metadata['difficulty']
    
    difficulty_inds = defaultdict(list)
    for k, v in difficulty_map.items():
        difficulty_inds[v].append(k)
    
    return difficulty_map, difficulty_inds
 
    
def get_res_test_cases(results):
    ret = []
    for res in results.values():
        res = res[0]
        ret.append([False if test_case == -1 or test_case == -2 else test_case for test_case in res])
    return ret

def count_compile_error(results):
    # counts results not existing as compile error
    count = 0
    ind = max(int(i) for i in results.keys())
    for i in range(ind+1):
        res = results.get(str(i), None)
        if not res:
            count += 1
        elif not res[0] or res[0][0] == -2: # compile error
            count += 1
    return count 


def count_success(results):
    count = 0
    total = 0
    avgs = []
    for res in results.values():
        res = res[0] # we care about the results in one sample so the zero index4
        if len(res) > 0:
            temp_count = sum(1 for test_case in res if test_case is True)
            count += temp_count
            total += len(res)
            avgs.append(temp_count/len(res))
    
    return count, total, np.mean(avgs)
    
    

def main(args):
    files = glob(join(args.results_folder, "*.jsonl"))
    results = {}
    for file in files:
        with jsonlines.open(file) as reader:
            results = {**results, **reader.read()}    
    max_ind = max(int(i) for i in results.keys())
    
    print(f'num results = {len(results)}')
    print(f'max index = {max_ind}')
    cnt_compile_error = count_compile_error(results)
    cnt_success, cnt_total, avg_success = count_success(results)
    percent_strict_success = np.mean([1 if all(test_cases) else 0 for test_cases in 
                                      get_res_test_cases(results)])
    
    metrics = {}
    
    metrics_all = {
        'count compile error': cnt_compile_error,
        'count passed test cases': cnt_success,
        'conut total test cases': cnt_total, 
        'count total problems': len(results),
        'percent strict success (over total problems)': percent_strict_success,
        'percent avg sucess (for each problem)': avg_success
    }
    
    metrics['all'] = metrics_all
    
    if args.test_path:
        diffculty_map, difficulty_inds = get_problems_difficulty(args.test_path)
        
        for difficulty, inds in difficulty_inds.items():
            results_filterd = {str(ind): results[str(ind)] for ind in inds if str(ind) in results}
            if len(results_filterd) == 0:
                continue
            cnt_success, cnt_total, avg_success = count_success(results_filterd)
            percent_strict_success = np.mean([1 if all(test_cases) else 0 for test_cases in 
                                      get_res_test_cases(results_filterd)])
            metrics_temp = {
                    'count passed test cases': cnt_success,
                    'count total test cases': cnt_total, 
                    'count total problems': len(results_filterd),
                    'percent strict success (over total problems)': percent_strict_success,
                    'percent avg sucess (for each problem)': avg_success
            }
            metrics[difficulty] = metrics_temp
    pprint(metrics)
    return metrics
    

    

            
    

if __name__ == '__main__':
    
    RESULTS_FOLDER = '/projects/bdata/kenqgu/Courses/CSE599_EMFML/apps/eval/results/codex_1_sample/test_results'
    APPS_TEST_PATH = '/projects/bdata/kenqgu/datasets/apps_dataset/test/'

    parser = argparse.ArgumentParser(description="Parsing the results folder. Expects a folder containing jsonl files, one for each problem in apps")
    parser.add_argument("--results_folder", default=RESULTS_FOLDER, type=str, help="path to the folder containing the results of the solutions")
    parser.add_argument("--test_path", default=APPS_TEST_PATH, type=str, help="path to the test folder of the app dataset")
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int, help="If you want to evaluate a subset of problems specify start and ending index. File with start and ending prefix must exist typically used with batch evaluation.")
    parser.add_argument("-i", "--index", default=0, type=int)
    
    args = parser.parse_args()
    
    main(args)