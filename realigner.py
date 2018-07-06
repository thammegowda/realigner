# Author :  Thamme Gowda ;; Created : July 04, 2018

import argparse
import glob
import itertools
import logging as log
import os
import sys
import lxml.etree as et
from collections import OrderedDict
from typing import List, Tuple, Optional
import multiprocessing as mp

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from ltfreader import read_ltf_doc, Doc
from scorer import get_scorer

log.basicConfig(level=log.INFO)
debug_mode = False


class Alignment:

    def __init__(self, src_id, tgt_id, alignments: List[Tuple[List[str], List[str], float]]):
        self.src_id = src_id
        self.tgt_id = tgt_id
        self.alignments = alignments


def read_doc_id_mapping(aln_file):
    aln_el = et.parse(aln_file).getroot()
    return aln_el.attrib["source_id"], aln_el.attrib["translation_id"]


def read_doc_alignments(aln_dir):
    aln_files = glob.glob(f'{aln_dir}/*.aln.xml')

    for i, f in enumerate(aln_files):
        if debug_mode and i == 50:
            log.warning("Aborting early in debug mode")
            break
        yield read_doc_id_mapping(f)


def write_alignment(path: str, aln: Alignment):
    log.info(f"Writing alignment file {path}")
    root = et.Element("alignments")
    root.attrib['source_id'] = aln.src_id
    root.attrib['translation_id'] = aln.tgt_id
    tree = et.ElementTree(root)
    for source_ids, target_ids, score in aln.alignments:
        alignment = et.Element("alignment")
        alignment.attrib['score'] = f'{score:.4f}'
        source = et.Element("source")
        source.attrib['segments'] = " ".join(source_ids)
        trans = et.Element("translation")
        trans.attrib['segments'] = " ".join(target_ids)
        alignment.append(source)
        alignment.append(trans)
        root.append(alignment)
    tree.write(path, pretty_print=True)


def re_align_segs(src_doc: Doc, eng_doc: Doc, scorer, threshold=0.0) -> Optional[Alignment]:

    srcs, tgts = src_doc.get_segs(), eng_doc.get_segs()
    scores = {}
    for (src_sid, src_txt), (tgt_sid, tgt_txt) in itertools.product(srcs, tgts):
        # TODO: skip if texts are not compatible
        scores[(src_sid, tgt_sid)] = scorer.score(src_txt, tgt_txt)

    # Rev sort by scores
    items = [entry for entry in scores.items() if entry[1] >= threshold]
    items = sorted(items, key=lambda x: x[1], reverse=True)
    fwd_matching, rev_matching = OrderedDict(), {}
    for (id1, id2), score in items:
        if id1 not in fwd_matching and id2 not in rev_matching:
            fwd_matching[id1] = id2, score
            rev_matching[id2] = id1, score
    if debug_mode:
        src_sids, tgt_sids = zip(*scores.keys())
        missed_src = src_sids - fwd_matching.keys()
        if missed_src:
            log.debug(f'Document: {src_doc.doc_id} missed alignments for {missed_src}')
        missed_tgt = tgt_sids - rev_matching.keys()
        if missed_tgt:
            log.debug(f'Document: {eng_doc.doc_id} missed alignments for {missed_tgt}')
    assert len(fwd_matching) == len(rev_matching)
    if not fwd_matching:
        return None
    if debug_mode:
        log.debug(f"Match {src_doc.doc_id} x {eng_doc.doc_id}")
        for src_sid, (tgt_sid, score) in fwd_matching.items():
            log.debug(f"MATCH :: {score} SRC: {src_doc.get_seg(src_sid)} \t\t TGT: {eng_doc.get_seg(tgt_sid)}")
    aligns = [([src_sid], [tgt_sid], score) for src_sid, (tgt_sid, score) in fwd_matching.items()]
    return Alignment(src_doc.doc_id, eng_doc.doc_id, aligns)


class ReAlignTask:
    """For multi processing"""
    def __init__(self, found_dir, out_dir, scorer, threshold):
        self.out_dir = out_dir
        self.found_dir = found_dir
        self.scorer = scorer
        self.threshold = threshold

    def ltf_path(self, doc_id):
        lang = doc_id.split('_')[0].lower()
        return f'{self.found_dir}/{lang}/ltf/{doc_id}.ltf.xml'

    def aln_path(self, doc_id):
        return f'{self.out_dir}/{doc_id}.aln.xml'

    def run(self, ids):
        """ For the sake of prarallelization"""
        src_id, eng_id = ids
        if src_id.lower().startswith('eng'):  # if swapping needed
            src_id, eng_id = eng_id, src_id
        out_path = self.aln_path(src_id)
        if os.path.exists(out_path):
            log.info(f'Skip: {src_id} x {eng_id} :: File exists {out_path}')
            return
        log.info(f"Going to align {src_id} x {eng_id}")
        src_doc = read_ltf_doc(self.ltf_path(src_id))
        eng_doc = read_ltf_doc(self.ltf_path(eng_id))
        new_algn = re_align_segs(src_doc, eng_doc, self.scorer, self.threshold)
        if new_algn:
            write_alignment(out_path, new_algn)
        else:
            log.warning(f'{src_id} x {eng_id} :: No alignment possible')


def re_align_all(doc_mapping: List[Tuple[str, str]], found_dir, out_dir, scorer, threshold, threads=2):
    log.info(f"Going to use {threads} threads")
    task_pool = mp.Pool(threads)
    task = ReAlignTask(found_dir, out_dir, scorer, threshold)
    task_pool.map(task.run, doc_mapping)
    task_pool.close()
    task_pool.join()


def main(found_dir, src_lang, out_dir, flags, **args):
    subs = os.listdir(found_dir)
    assert 'eng' in subs
    assert src_lang in subs
    assert 'sentence_alignment' in subs

    aln_dir = f'{found_dir}/sentence_alignment'
    if '/' not in out_dir:
        out_dir = f'{found_dir}/{out_dir}'
    log.info(f"Output dir {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    aln_maps = list(read_doc_alignments(aln_dir))
    log.info(f"Found {len(aln_maps)} doc mappings")
    scorer = get_scorer(flags, debug=debug_mode, max_vocab=args.pop('max_vocab'),
                        src_emb=args.pop('src_emb'), eng_emb=args.pop('eng_emb'))
    re_align_all(aln_maps, found_dir=found_dir, out_dir=out_dir, scorer=scorer, **args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-fd', '--found-dir', type=str, required=True,
                   help='Path to "found" dir that has eng and xyz lan')
    p.add_argument('-l', '--lang', dest='src_lang', type=str, required=True, help='source language code')
    p.add_argument('-o', '--out-dir', type=str, default='sentence_alignment-tg')
    p.add_argument('-se', '--src-emb', type=str, help='path to source language embedding (MCSS vectors)')
    p.add_argument('-ee', '--eng-emb', type=str, help='path to english language embedding (MCSS vectors)')
    p.add_argument('-th', '--threshold', type=float, default=0.0,
                   help='threshold score below which the sentence pairs must be ignored')
    p.add_argument('-nt', '--threads', type=int, default=2, help='Number of threads to use')
    p.add_argument('-mv', '--max-vocab', type=int, default=int(1e6), help='Maximum Vocabulary size (MCSS vectors)')
    p.add_argument('-f', '--flags', type=str, default='charlen,toklen,copypatn,ascii,mcss',
                   help='comma separated list of scorers to use. For example set -f "mcss" to use only MCSS or'
                        ' "copypatn,mcss" to use copy pattern scorer and MCSS')
    p.add_argument('-d', '--debug', action='store_true', help="Turn on the debug mode")
    args = vars(p.parse_args())
    if args.pop('debug'):
        log.getLogger().setLevel(level=log.DEBUG)
        debug_mode = True
        log.debug("Debug Mode ON")
    main(**args)

