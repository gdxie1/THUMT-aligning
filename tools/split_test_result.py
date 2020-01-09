#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import json
import codecs
import operator
import cPickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="split Translated test file",
        usage="split_test_result.py [<args>] [-h | --help]"
    )

    # input file
    parser.add_argument("--input", type=str, default= None, required=False,
                        help="Path of input corpus")

    parser.add_argument("--output", type=str, default=None, required=False,
                        help="Path to output")
    return parser.parse_args()

def main(args):

    output_files=['nist03.result', 'nist04.result', 'nist05.result', 'nist06.result', 'nist08.result']
    line_index = [0, 919, 2707, 3789, 5453, 6810]
    with codecs.open(args.input, "r", encoding='utf8') as input_file:
        source_lines = [line for line in input_file]  # if line.strip()

    for i in range(5):
        output_list = source_lines[line_index[i]:line_index[i+1]]
        with codecs.open(args.output+'/'+output_files[i], "w", encoding='utf8') as o_file:
            for line in output_list:
                o_file.write(line)

if __name__ == "__main__":
    main(parse_args())