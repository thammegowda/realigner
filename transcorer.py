# Scorer of translations based on Translation tables

from typing import Dict, Set
from ttab import TTable
import argparse
import sys
import logging as log
import pickle

debug_mode = False


class TranScorer:
    """Translation scorer"""

    def __init__(self, ttab: TTable, combine='sum'):
        """

        :param ttab: TTable instance
        :param combine: way to combine evidence, sum or max
        """
        self.src_tgt = ttab.fwd
        self.tgt_src = ttab.inv
        self.src_prep = ttab.src_prep
        self.tgt_prep = ttab.tgt_prep
        self.combiner = {'sum': sum, 'max': max}[combine]
        log.info(f"Translation Scorer source: {ttab.src} tgt:{ttab.tgt}, combiner ={combine}")

    @classmethod
    def new(cls, ttab_path):
        with open(ttab_path, 'rb') as f:
            log.info(f"Loading TTable from {ttab_path}")
            ttab = pickle.load(f)
        return cls(ttab)

    def _translation_evidence(self, tok: str, ttab: Dict[str, Dict[str, float]], cand_toks: Set[str]):
        """
        :param tok source token that may have generated cand_toks
        :param ttab: translation probability table
        :param cand_toks: Candidate tokens which we are suspected to be generated by input tok
        :return:  0.0 <= score <= 1.0
        """

        # token generation probability distribution. Sum of values should be summed to 1.0.
        # this should be P(cand_toks | token)
        probs = ttab.get(tok)
        if not probs:   # tok is an OOV
            if tok in cand_toks:
                return 1.0  # tok was copied over
            # OOV Nothing we can do about it ; we dont know what happened there
            # Maybe romanize tokens and see if name matches
            return 0.0

        # find the candidate tokens that may be generated from distribution and sum them up
        cand_scores = [probs[tok] for tok in probs.keys() & cand_toks]
        return self.combiner(cand_scores) if cand_scores else 0.0

    def score(self, src, tgt):
        src_toks = self.src_prep(src)
        tgt_toks = self.tgt_prep(tgt)
        # NOTE: in this version, the repeated use of tokens are not dealt with
        src_tok_set, tgt_tok_set = set(src_toks), set(tgt_toks)

        # Source token generating these tokens comes from normal ttab :: P(tgt | src) i.e. src-to-tgt
        src_tok_usage = [self._translation_evidence(src_tok, self.src_tgt, tgt_tok_set) for src_tok in src_toks]
        # Target tokens generating the given sentence ;; these come from inverse ttab :: P(tgt | src) i..e tgt-to-src
        tgt_tok_usage = [self._translation_evidence(tgt_tok, self.tgt_src, src_tok_set) for tgt_tok in tgt_toks]

        src_evidence = sum(src_tok_usage) / len(src_tok_usage)
        tgt_evidence = sum(tgt_tok_usage) / len(tgt_tok_usage)
        if debug_mode:
            src_data = ' '.join(map(lambda r: f'{r[0]}:{r[1]:.4f}', zip(src_toks, src_tok_usage)))
            tgt_data = ' '.join(map(lambda r: f'{r[0]}:{r[1]:.4f}', zip(tgt_toks, tgt_tok_usage)))
            log.debug(f'SRC:: {src_evidence:.4f} :: {src_data}')
            log.debug(f'TGT:: {tgt_evidence:.4f} :: {tgt_data}')

        return (src_evidence + tgt_evidence) / 2.0

    def score_all(self, records, parse=True):
        if parse:
            records = (r.split('\t') for r in records)
        for src, tgt in records:
            yield self.score(src, tgt), src, tgt


def main(scorer, inp, out, **args):
    for score, src, tgt in scorer.score_all(inp):
        out.write(f'{score:.4f}\t{src}\t{tgt}\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path. Format: Source<tab>Target sentence per line')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path')
    p.add_argument('-t', '--ttab', dest='ttab_path', type=str, required=True,
                   help='Translation Table file (pickle dumb of ttab.TTable object, see ttab.py to get one)')

    p.add_argument('--test', action='store_true',
                   help="Turn on the test mode. In test mode, assume the input is parallel text "
                        "(i.e. positive alignments) and randomly shuffles the input to obtain negative alignments")
    p.add_argument('-n', '--neg-samples', dest='neg_sample_count', type=int, default=40,
                   help='Number of random negative samples to test against (in --test mode)')
    p.add_argument('-s', '--seed', type=int, default=None,
                   help='seed for reproducing random shuffle for negatives (in --test mode)')
    args = vars(p.parse_args())
    scorer = TranScorer.new(args.pop('ttab_path'))
    if args.pop('test'):
        from utils import scorer_eval
        scorer_eval(scorer, **args)
    else:
        main(scorer, **args)
