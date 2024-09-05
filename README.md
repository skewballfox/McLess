# McLess

Multi-class Least Error Squared Sum

## Introduction

This is a basic implementation of the algorithm (described in [this paper](https://skim.math.msstate.edu/reprints/Park-ETAL-mCLESS-CSCE-2023.pdf)), along with some code for benchmarking it against other classification algorithms. 

Originally this was an assignment for numerical linear algebra so I'd like to thank dr. Seongjai Kim, for making it possible for me to remotely understand how this works(and thus how to implement it).

## Setting up the development environment

```sh
git clone https://github.com/skewballfox/McLess
cd McLess
poetry install
```


to run the benchmark (with venv activated)
```
python test/classifier_bench.py
```
