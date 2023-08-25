# Blackbird
Blackbird is a dataset of synthetic relational puzzles, inspired by the RAVEN dataset, created to demonstrate the reasoning capabilities of the quantum model of concepts.

This repository contains the Python code used to generate two versions of the dataset:
- **Balanced dataset** In this dataset, both the number of correct puzzles and correct constraints
are balanced. With a probability of 0.5, a puzzle is generated that satisfies either all constraints
or none of the constraints. Therefore, the correctness of the constraints is dependent on
each other.
- **Independent dataset** The correctness in the constraints is independent of each other in this
dataset. This comes at the cost of class imbalance for either the puzzles or the constraints.
We try to make these two imbalances equally small by choosing the probability of generating
an incorrect constraint to be p = 0.22 such that
$P(\text{correct puzzle}) = (1 − p)^6 \approx p = P(\text{incorrect constraint})$.

In addition, two pre-generated datasets are provided containing 3000 training, 300 validation and 300 testing instances each.
