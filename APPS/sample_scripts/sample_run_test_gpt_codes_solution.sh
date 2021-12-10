conda activate TransformerCodeComparison_GPU
for i in {0..4999} ; do 
  python3 /projects/bdata/kenqgu/Courses/CSE599_EMFML/apps/eval/test_one_solution.py -t /projects/bdata/kenqgu/Courses/CSE599_EMFML/apps/train/test.json --root=/projects/bdata/kenqgu/datasets/apps_dataset/test --save=/projects/bdata/kenqgu/Courses/CSE599_EMFML/apps/eval/results/gpt2_1_sample/all_codes_05-12-2021_08-38-11.jsonl -i $i --jsonl;
done
