# TransformersCodeEval

This repo includes code and scripts to evaluate Transformers on two code datasets namely [APPS](https://github.com/hendrycks/apps) and [PythonProgrammingPuzzles](https://github.com/microsoft/PythonProgrammingPuzzles). 

We evauate on GPT-2 family of transformers as well as OpenAI's codex model. Everything was ran on Python 3.7. The libraries used are in `requirements.txt`

For further instructions go to each respective dataset's README. 

## APPS

The APPs dataset aims to evaluate how well models are able to generate syntactically and functionally correct code given specific task directions. Each example consists of natural language task specification and sample input. Evaluation is done by observing whether the program passes a suite of test cases. In total there are 10,000 examples 5000 for training and 5000 for testing. Examples are manually curated from websites in which programmers share problems with each other.

## PythonProgramingPuzzles

The Programming Puzzles dataset aims to evaluate how a model can write code to tackle puzzles such as the "Tower of Hanoi" problem and use algorithmic tools such as recursion. 

For example
```python
# Tower of Hanoi, often teaches recursion. Move [i, j] means move top disk on tower i to j, with 1 ≤ i,j ≤ 3
def f2(moves: List[List[int]], num_disks=8):
    state = [1] * num_disks # All disks start at tower 1.
    for [i, j] in moves:
        assert state.index(i) <= (state + [1, 2, 3]).index(j), "bigger disk on top"
        state[state.index(i)] = j # Move smallest disk from tower i to tower j.
    return state == [3] * num_disks # All disks must end on tower 3.

```

The authors argue that imrpovement in solving puzzles may lead to performance in other tasks. 