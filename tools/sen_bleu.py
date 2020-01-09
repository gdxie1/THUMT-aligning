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
#import xml.etree.ElementTree as ET
import re
import collections


# 从BLEU打分结果中，对比每一句的打分结果

def parse_args():
    parser = argparse.ArgumentParser(
        description="generate context file",
        usage="gen_context.py [<args>] [-h | --help]"
    )

    # input file
    parser.add_argument("--src", type=str, default=None, required=True,
                        help="Path of source file")
    parser.add_argument("--trans", type=str, default=None, required=False,
                        help="Path to translated sgm file")
    parser.add_argument("--trans_better", type=str, default=None, required=False,
                        help="Path to better translated sgm file")
    parser.add_argument("--refs", type=str, default=None, required=False,
                        help="Path to ref sgm file")
    parser.add_argument("--bleu_basic", type=str, default=None, required=False,
                        help="bleu score of the basic model")
    parser.add_argument("--bleu_better", type=str, default=None, required=False,
                        help="improved bleu score result")


    parser.add_argument("--output", type=str, default=2, required=False,
                        help="Path to output")
    return parser.parse_args()


def read_bleu_to_dict(bleu_file):

    with codecs.open(bleu_file, "r", encoding='utf8') as input_file:
        bleu_basic_lines = [line for line in input_file if line.strip()]

    bleu_list = collections.OrderedDict()

    bleu_re = re.compile(r'score using 4-grams =(.*) for system')
    sen_id_re = re.compile(r'segment (.*) of document')
    doc_id_re = re.compile(r'of document "(.*)"')
    for line in bleu_basic_lines:
        while line[0] == u' ':  # strip can't remove the space at the head part
            line = line[1:]
        if line.startswith("BLEU score using 4-grams ="):
            bleu_v = bleu_re.search(line).group(1)
            sen_id = sen_id_re.search(line).group(1)
            doc_id = doc_id_re.search(line).group(1)

            bleu_v = float(bleu_v)

            if doc_id in bleu_list:
                doc_dict = bleu_list[doc_id]  # if existed, then get it
            else:
                doc_dict = collections.OrderedDict()  # if first time, then create a ordered dict

            id_sen_list = doc_dict.get(sen_id, [])

            id_sen_list.append(bleu_v)
            doc_dict[sen_id] = id_sen_list
            bleu_list[doc_id] = doc_dict

    return bleu_list

def read_sgm_to_dict(sgm_file):

    TAG_RE = re.compile(r'<[^>]+>')
    DOCID_RE = re.compile(r'<seg id=[0-9]+>')

    trans_list = collections.OrderedDict()

    # doc_sen_list = []  # sentences belong to a doc
    # doc_senid_list = []  # sentences belong to a doc
    doc_id = ''
    doc_start_flag = False
    with codecs.open(sgm_file, "r", encoding='utf8') as input_file:
        source_lines = [line for line in input_file if line.strip()]
    for line in source_lines:
        if line.startswith("<doc docid=") or line.startswith("<DOC docid="):
            endpos = line.find('\"', 12)
            doc_id = line[12:endpos]
            # if doc_id == "AFC20030102.0015":
            #     a = 1
            if doc_id in trans_list:
                doc_dict = trans_list[doc_id]  # if existed, then get it
            else:
                doc_dict = collections.OrderedDict()  # if first time, then create a ordered dict

            doc_start_flag = True  # 防止格式出错，

        if line.startswith("<seg id="):
            # extract the sen id
            matched_doc_id = DOCID_RE.match(line)
            index = matched_doc_id.regs[0]
            sen_id = line[index[0]:index[1]]
            sen_id = sen_id[8:-1]

            id_sen_list = doc_dict.get(sen_id, [])

            # extract the sentence
            id_sen_list.append(TAG_RE.sub('', line))
            doc_dict[sen_id] = id_sen_list

        if line.startswith("</DOC"):
            trans_list[doc_id] = doc_dict  # each documental id corresponding a dict of sentence with sen is as key
            assert doc_start_flag
            doc_start_flag = False

    return trans_list


def main(args):

    with codecs.open(args.src, "r", encoding='utf8') as input_file:
        source_lines = [line for line in input_file if line.strip()]
    total_line = len(source_lines)


    bleu_basic = read_bleu_to_dict(args.bleu_basic)
    bleu_better = read_bleu_to_dict(args.bleu_better)

    #total_line = len(source_lines)
    translated_sgm = read_sgm_to_dict(args.trans)
    refs_sgm = read_sgm_to_dict(args.refs)


    # convert translated to list
    translated_items = []
    for docid, doc_dict in translated_sgm.items():
        for sen_id, sen_list in doc_dict.items():
            translated_items.append((docid, sen_id, sen_list[0]))

    filename = args.src[args.src.rfind("/"):-1]
    with codecs.open(args.output + '/' + filename + '.compare',
                     "w", encoding='utf8') as o_file:
        for i, src_sen in enumerate(source_lines):
            print(i)
            o_file.write(src_sen)
            o_file.write(translated_items[i][2])
            doc_dict = refs_sgm[translated_items[i][0]]  # get doc dict from doc id
            sen_list = doc_dict[translated_items[i][1]]   # get sen dict from sen id

            bleu_basic_doc_dict = bleu_basic[translated_items[i][0]]  # get doc dict from doc id
            bleu_basic_sen_list = bleu_basic_doc_dict[translated_items[i][1]]  # get sen dict from sen id

            bleu_better_doc_dict = bleu_better[translated_items[i][0]]  # get doc dict from doc id
            bleu_better_sen_list = bleu_better_doc_dict[translated_items[i][1]]  # get sen dict from sen id

            o_file.write("%f vs %f\n" % (bleu_basic_sen_list[0], bleu_better_sen_list[0]))

            o_file.write("--------\n")
            for sen in sen_list:
                o_file.write(sen)
            o_file.write("------------------\n")



if __name__ == "__main__":
    main(parse_args())
