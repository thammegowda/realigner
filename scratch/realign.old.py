#!/usr/bin/env python3

""""
Sentence re aligner tool
"""

import argparse
import sys
from typing import List, Iterator, Tuple, Optional
from collections import OrderedDict, namedtuple
import logging as log
log.basicConfig(level=log.INFO)

debug_mode = log.getLogger().isEnabledFor(level=log.DEBUG)
Match = namedtuple('Match', ['src', 'tgt', 'score'])
AlignedDoc = List[Match]


class Document:

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.src_ids = set()
        self.tgt_ids = set()
        self.scores_map = {}

    def add_rec(self, src_id: str, tgt_id: str, score: float):
        self.src_ids.add(src_id)
        self.tgt_ids.add(tgt_id)
        self.scores_map[src_id, tgt_id] = score


def read_documents(inp: Iterator[Tuple[str, str, float]]) -> Iterator[Document]:
    """
    converts a stream of lines into documents
    :param inp:
    :return:
    """
    doc = None
    i = 0
    for src_id, tgt_id, score in inp:
        doc_id = src_id.split('.')[0]
        if doc is None:
            doc = Document(doc_id)
        if doc.doc_id != doc_id:
            yield doc
            doc = Document(doc_id)
            i = 0
        doc.add_rec(src_id, tgt_id, score)
        i += 1
    if doc:
        yield doc


def rematch(doc: Document, threshold: Optional[float]=None) -> AlignedDoc:
    """
    Rematch sentences within document based on the scores
    :param doc: document to be re-matches
    :param threshold: threshold value of score, documents with score lesser than this are ignored
    :return: an AlignedDoc
    """
    # Sort by scores
    items = sorted(doc.scores_map.items(), key=lambda x: x[1], reverse=True)
    if threshold is not None:
        items = [entry for entry in items if entry[1] >= threshold]
    fwd_matching, rev_matching = OrderedDict(), {}
    for (id1, id2), score in items:
        if id1 not in fwd_matching and id2 not in rev_matching:
            fwd_matching[id1] = id2, score
            rev_matching[id2] = id1, score
    if debug_mode:
        missed_src = doc.src_ids - fwd_matching.keys()
        if missed_src:
            log.debug(f'Document: {doc.doc_id}, Source side missed alignment for {missed_src}')
        missed_tgt = doc.tgt_ids - rev_matching.keys()
        if missed_src:
            log.debug(f'Document: {doc.doc_id}, Target side missed alignment for {missed_tgt}')
    assert len(fwd_matching) == len(rev_matching)
    res = [Match(src, tgt, score) for src, (tgt, score) in fwd_matching.items()]
    return res


def realign_segments(recs, threshold: float, parse=True):
    """

    :param recs: input records
    :param threshold: threshold to ignore low scored records
    :param parse: if the input is text stream. set to false if input is Iterator[Tuple[str, str, float]]
    :return: filtered records
    """
    def parser(line):
        src_id, tgt_id, score = map(lambda x: x.strip(), line.strip().split('\t'))
        return src_id, tgt_id, float(score)

    recs = map(parser, recs) if parse else recs
    docs = read_documents(recs)
    aln_docs = (rematch(doc, threshold) for doc in docs)
    count = 0
    for doc in aln_docs:
        count += 1
        yield from doc
    log.info(f"Processed {count} docs")


def main(inp, out, threshold):
    line_count = 0
    for rec in realign_segments(inp, threshold):
        line = f'{rec.src}\t{rec.tgt}\t{rec.score:.4f}\n'
        out.write(line)
        line_count += 1
    log.info(f"Wrote {line_count} segments to output")


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file having one record per line. Format: doc_id.seg_id<tab>doc_id.seg_id<tab>score.'
                        'Score must be a floating point number, higher value signifies better match.')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file')
    p.add_argument('-t', '--threshold', type=float, default=0.0,
                   help='Threshold value: ignore records with scores lower than this')
    p.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    args = vars(p.parse_args())
    if args.pop('verbose'):
        log.getLogger().setLevel(level=log.DEBUG)
        debug_mode = True

    main(**args)
