#!/usr/bin/env python
# Author: Thamme Gowda ; Created: July 07, 2018

"""
Translation Table Data Structure
"""
import logging as log
import re
import pickle
from typing import Dict
from collections import OrderedDict


class TTable:
    """
    Translation Table - alignment information from Giza Aligner
    """

    def __init__(self, src: str, tgt: str, src_vocab: str, tgt_vocab: str, fwd_table: str, inv_table: str=None):
        """
        creates a translational table
        :param src: source language code
        :param tgt: target language code
        """
        self.src = src
        self.tgt = tgt
        log.info(f"Vocabulary Files: {src}: {src_vocab};  {tgt}:{tgt_vocab}")
        self.src_id2tok, self.src_freq = TTable.load_vocab(src_vocab)
        self.tgt_id2tok, self.tgt_freq = TTable.load_vocab(tgt_vocab)
        self.src_tok2id, self.tgt_tok2id = TTable.reverse_map(self.src_id2tok), TTable.reverse_map(self.tgt_id2tok)
        log.info("Vocabulary Size: SRC: %d; TGT:%d" % (len(self.src_id2tok), len(self.tgt_id2tok)))

        self.fwd: Dict[str, OrderedDict[str, float]] = self.read_ttab(fwd_table, self.src_id2tok, self.tgt_id2tok)
        self.inv: Dict[str, OrderedDict[str, float]] = \
            self.read_ttab(inv_table, self.tgt_id2tok, self.src_id2tok) if inv_table else {}
        log.info("T-Tab Size: Normal: %d; inverse:%d" % (len(self.fwd), len(self.inv)))

    def store_at(self, path):
        log.info('storing at %s' % path)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def vocab_match(self, pattern, source=True):
        # TODO: use a trie to support prefix match
        vocab = self.src_tok2id.keys() if source else self.tgt_tok2id.keys()
        yield from (key for key in vocab if key and re.match(pattern, key))

    @staticmethod
    def reverse_map(data, one_to_one=True):
        rev = {v: k for k, v in data.items()}
        if one_to_one:
            assert len(rev) == len(data)
        return rev

    @staticmethod
    def load_from(path):
        log.info("Loading from %s" % path)
        ttab = pickle.load(open(path, 'rb'))
        assert type(ttab) is TTable
        log.info("Vocabulary Size: SRC: %d; TGT:%d" % (len(ttab.src_id2tok), len(ttab.tgt_id2tok)))
        log.info("T-Tab Size: Normal: %d; Inverse:%d" % (len(ttab.fwd), len(ttab.inv)))
        return ttab

    @staticmethod
    def load_vocab(path, augment=((0, None, None),)):
        """
        loads vocabulary
        :param path: path of vocabulary
        :param augment: additional data to be augmented. Example = [(0, None), (1, 'UNK')]
        :return: dict, dict - the first one has id to token mapping, the second one has id to frequency
        """
        with open(path) as f:
            id2tok = {}
            freq = {}
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                idx, tok, count = line.split()
                idx, count = int(idx), int(count)
                id2tok[idx] = tok
                freq[idx] = count
            if augment:
                for idx, tok, count in augment:
                    assert idx not in id2tok
                    id2tok[idx] = tok
                    freq[idx] = count
            return id2tok, freq

    @staticmethod
    def read_ttab(path, idx1, idx2) -> Dict[str, Dict[str, float]]:
        ttab = {}
        with open(path) as f:
            for line in f:
                src_id, tgt_id, prob = line.split()
                src_id, tgt_id, prob = int(src_id), int(tgt_id), float(prob)
                src_tok, tgt_tok = idx1[src_id], idx2[tgt_id]
                if src_tok not in ttab:
                    ttab[src_tok] = OrderedDict()
                ttab[src_tok][tgt_tok] = prob
        return ttab


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='T-Table compressor', )
    parser.add_argument('-s', '--src', required=True, type=str, help='Source language code, example: esp')
    parser.add_argument('-t', '--tgt', type=str, default='eng', help='Target language code, example: eng')
    parser.add_argument('-ft', '--fwd-table', required=True, type=str,
                        help='Forward table from Giza. example: GIZA.normal.t3.final')
    parser.add_argument('-it', '--inv-table', type=str,
                        help='Inverse table from Giza. example: GIZA.invers.t3.final')

    parser.add_argument('-sv', '--src-vocab', required=True, type=str,
                        help='Source vocabulary file. Format: Index<space>Word<space>Count per line')
    parser.add_argument('-tv', '--tgt-vocab', required=True, type=str,
                        help='Target vocabulary file.  Format: Index<space>Word<space>Count per line')

    parser.add_argument('-o', '--out', help='Store the compressed T-Tab at this path', required=True)
    args = vars(parser.parse_args())
    out = args.pop('out')
    ttab = TTable(**args)
    if not out.endswith('.pkl') and not out.endswith('.pickle'):
        out += '.pkl'
    ttab.store_at(out)
