#!/usr/bin/env sh

pipenv run check-pdf test-data/dna-paper.pdf > test-data/baselines/dna-paper.txt
