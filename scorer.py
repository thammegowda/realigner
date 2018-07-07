# Author :  Thamme Gowda ;; Created : July 04, 2018
"""
A scorer for mining parallel/comparable sentences from comparable documents
"""
import logging as log
import re

log.basicConfig(level=log.INFO)


class UnifiedScorer:

    must_reject = -20
    must_accept = +20
    may_reject = -1
    may_accept = +1
    not_sure = 0
    final_pos_score = 1.0
    final_neg_score = -1.0

    copy_patterns = [re.compile(p) for p in [r'(\d+)', r'(https?://[^ ]+)']]

    def __init__(self, final_scorer, flags='charlen,toklen,copypatn,ascii', debug=False):
        flags = flags.split(',') if type(flags) is str else flags
        mapping = {
            'copypatn': self.copy_score,
            'toklen': self.toklen_score,
            'charlen': self.charlen_score,
            'ascii': self.ascii_ratio_score,
        }
        flags = flags.split(',') if type(flags) is str else flags
        self.scorers = [mapping[flag] for flag in flags]
        if not final_scorer:
            log.warning('Final Scorer is None, this setting is not recommended')
        self.final_scorer = final_scorer.score if final_scorer else None
        self.debug = debug

    def copy_score(self, src: str, tgt: str) -> float:
        score = 0.0
        for pat in self.copy_patterns:
            src_toks = set(pat.findall(src))
            tgt_toks = set(pat.findall(tgt))
            is_aligned = len(src_toks) == len(tgt_toks) == len(src_toks & tgt_toks)
            if not is_aligned:
                score += self.must_reject
                log.debug(f"Going to reject, because SRC: {src_toks} TGT:{tgt_toks} are not same")
            else:
                # number of matches used as evidence. zero matches must yield zero score
                # score += 2 * self.may_accept * len(src_toks)
                # EDIT: going to accept straight away if there is atleast one match
                score += self.must_accept * len(src_toks)
        return score

    def charlen_score(self, src: str, tgt: str) -> float:
        ratio = (1 + len(src)) / (1 + len(tgt))
        if 0.33 <= ratio <= 3.0:
            return self.not_sure
        else:
            return self.must_reject

    def toklen_score(self, src: str, tgt: str) -> float:
        ratio = (1 + len(src.split())) / (1 + len(tgt.split()))
        if 0.33 <= ratio <= 3.0:
            return self.not_sure
        else:
            return self.must_reject

    def ascii_ratio_score(self, src: str, tgt: str) -> float:
        """Check that there are approximately same ratio of punctuations and numerals (2x). exclude alphabets"""
        src_ct = 1.0 + sum(1 for c in src if ord(c) < 256 and not c.isalpha())
        tgt_ct = 1.0 + sum(1 for c in tgt if ord(c) < 256 and not c.isalpha())
        if 0.33 <= src_ct / tgt_ct <= 3.0:
            return self.may_accept
        else:
            return self.must_reject

    def score(self, src: str, tgt: str) -> float:
        # negative means No, positive means yes
        tot_score = 0.0
        for scorer in self.scorers:
            tot_score += scorer(src, tgt)
            if tot_score >= self.must_accept or tot_score <= self.must_reject:
                break  # abort the scoring here

        if tot_score >= self.must_accept:
            final_score = self.final_pos_score
        elif tot_score <= self.must_reject:
            final_score = self.final_neg_score
        else:
            final_score = self.final_scorer(src, tgt) if self.final_scorer else self.not_sure
            assert self.final_neg_score <= final_score <= self.final_pos_score
        if self.debug:
            log.debug(f'score:{tot_score:.4f} {final_score:.4f} :: SRC: {src} \t\t TGT: {tgt}')
        return final_score


class ScoreAggregator:
    """Aggregates scores from multiple scoring functions"""

    def __init__(self, scorers):
        assert len(scorer) > 0
        self.scorers = scorers

    def score(self, src, tgt):
        scores = [s.score(src, tgt) for s in self.scorers]
        return sum(scores) / len(scores)


def get_scorer(flags, debug=False, **args):
    scorers = []
    flags = flags.split(',')
    if 'mcss' in flags:
        flags.remove('mcss')
        src_emb, eng_emb, max_vocab = args.pop('src_emb'), args.pop('eng_emb'), args.pop('max_vocab')
        assert src_emb and eng_emb, '--src-emb and --eng-emb args are required if "mcss" is enabled'
        from mcss import MCSS
        scorers.append(MCSS(src_vec_path=src_emb, tgt_vec_path=eng_emb, nmax=max_vocab))
    if 'ttab' in flags:
        flags.remove('ttab')
        ttab_file = args.pop('ttab_file')
        assert ttab_file, '--ttab is needed for this combination of args'
        from transcorer import TranScorer
        scorers.append(TranScorer.new(ttab_file))

    final_scorer = None
    if len(scorers) == 1:
        final_scorer = scorers[0]
    elif len(scorer) > 1:
        final_scorer = ScoreAggregator(scorers)
    return UnifiedScorer(final_scorer, flags=flags, debug=debug) if flags else final_scorer


def predict(scorer, inp, out, **args):
    for line in inp:
        src, tgt = line.strip().split('\t')
        score = scorer.score(src, tgt)
        out.write(f'{score:.4f}\t{src}\t{tgt}\n')


if __name__ == '__main__':
    import argparse
    import sys
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Source<tab>english sentence per line')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout)
    p.add_argument('-f', '--flags', type=str,
                   help='list of scorers to use. "mcss" to use only the mcss scorer, or "ttab" to use just TTab scorer'
                        ' or "mcss,ttab" to use both',
                   default='charlen,toklen,copypatn,ascii,mcss,ttab')
    p.add_argument('-se', '--src-emb', type=str, help='path to source language embedding (flag=mcss)')
    p.add_argument('-ee', '--eng-emb', type=str, help='path to english language embedding (flag=mcss)')
    p.add_argument('-m', '--max-vocab', type=int, help='Max vocabulary size (flag=mcss)', default=int(1e6))
    p.add_argument('-tf', '--ttab-file', type=str, help='ttab.TTab pickle file (flag=ttab)')
    p.add_argument('-n', '--neg-samples', dest='neg_sample_count', type=int, default=40,
                   help='Number of random negative samples to test against')
    p.add_argument('-s', '--seed', type=int, default=None, help='seed for reproducing (random shuffle for negatives)')
    p.add_argument('-d', '--debug', action='store_true', help="Turn on the debug mode")
    p.add_argument('-t', '--test', action='store_true',
                   help="Turn on the test mode. In test mode, assume the input is parallel text "
                        "(i.e. positive alignments) and randomly shuffles the input to obtain negative alignments")
    args = vars(p.parse_args())
    scorer = get_scorer(**args)
    if args.pop('test'):
        from utils import scorer_eval
        scorer_eval(scorer, **args)
    else:
        predict(scorer, **args)
