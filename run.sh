#!/bin/bash
python main.py --latent 10 --lmbda 0 --train data/train.txt  --test data/test.txt --log roc.log --result roc.txt
# python main.py --latent 10 --lmbda 0 --train data/wordnet-mlj12-train.txt  --test data/wordnet-mlj12-test.txt --log roc.log --result roc.txt
# python main.py --latent 10 --lmbda 0 --train data/wordnet/wordnet-mlj12-train.txt  --test data/wordnet/wordnet-mlj12-test.txt --log wordnet.log --result result_wordnet.txt
