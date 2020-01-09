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

# 为清华大学的document NMT 生成context文件 即把前面的2句或多句链起来，形成一句

def parse_args():
    parser = argparse.ArgumentParser(
        description="generate context file separating with ###",
        usage="gen_context.py [<args>] [-h | --help]"
    )

    # input file
    parser.add_argument("--input", type=str, default=None, required=True,
                        help="Path of input corpus")
    parser.add_argument("--output_dir", type=str, default=None, required=False,
                        help="Path to output")
    parser.add_argument("--pre_sen_num", type=int, default=2, required=False,
                        help="Path to output")
    return parser.parse_args()

def main(args):

    with codecs.open(args.input, "r", encoding='utf8') as input_file:
        source_lines = [line for line in input_file if line.strip()]
    total_line = len(source_lines)
    output_list = []
    # 0 line
    context_line = "<eos>####"
    #context_line += "\n"
    output_list.append(context_line)

# ﻿train_src.1history

    for i in range(1, args.pre_sen_num):  # 1~ line
        context_line = ""
        for j in range(i-1, -1, -1):
            context_line = source_lines[j].rstrip()+'####'+context_line
        #context_line += "\n"
        output_list.append(context_line)

    for i in range(args.pre_sen_num,total_line):  # start from 1
        context_line = ""
        for j in range(i-args.pre_sen_num, i):
            context_line += source_lines[j].rstrip()+'####'
        #context_line += "\n"
        output_list.append(context_line)
    context_output = []
    for i in range(total_line):  # start from 1
        #context_line = ""
        context_line = output_list[i] + source_lines[i]
        context_output.append(context_line)

    filename_pos = args.input.rindex("/")
    input_name = args.input[filename_pos:]
    with codecs.open(args.output_dir + input_name + '.%dhistory' % (args.pre_sen_num+1),
                     "w", encoding='utf8') as o_file:
        for line in context_output:
            o_file.write(line)

if __name__ == "__main__":
    main(parse_args())