# Author :  Thamme Gowda ;; Created : July 04, 2018
"""
A scorer for mining parallel/comparable sentences from comparable documents
"""
import re
import logging as log
import random
from collections import defaultdict
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
        self.final_scorer = final_scorer
        if not final_scorer:
            log.warning('Final Scorer is None, this is not recommended')
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
                score += 2 * self.may_accept * len(src_toks)
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


def get_scorer(flags, max_vocab, src_emb=None, eng_emb=None, debug=False, **args):
    mcss = None
    flags = flags.split(',')
    if 'mcss' in flags:
        flags.remove('mcss')
        assert src_emb and eng_emb, '--src-emb and --eng-emb args are required if "mcss" is enabled'
        from mcss import MCSS
        mcss = MCSS(src_vec_path=src_emb, tgt_vec_path=eng_emb, nmax=max_vocab).score
    return UnifiedScorer(final_scorer=mcss, flags=flags, debug=debug) if flags else mcss


def predict(scorer, inp, out, **args):
    for line in inp:
        src, tgt = line.strip().split('\t')
        score = scorer.score(src, tgt)
        out.write(f'{score:.4f}\t{src}\t{tgt}\n')


def test(scorer, inp, out, neg_sample_count=20, **args):
    pos_exs = [line.strip().split('\t') for line in inp]
    pos_count = len(pos_exs)
    assert pos_count > neg_sample_count
    src_sqs, tgt_seqs = zip(*pos_exs)
    pos_scores = {src: scorer.score(src, tgt) for src, tgt in pos_exs}
    assert len(pos_scores) == pos_count
    error_tgts = defaultdict(set)
    neg_error = 0.0
    neg_count = 0
    for src, pos_tgt in pos_exs:
        neg_tgts = [t for t in tgt_seqs if t != pos_tgt]
        random.shuffle(neg_tgts)
        for neg_tgt in neg_tgts[:neg_sample_count]:
            pred_score = scorer.score(src, neg_tgt)
            if pred_score > pos_scores[src]:
                error_tgts[src].add((pred_score, neg_tgt))
            # assumption: negative score is definitely zero/neg class
            neg_error += max(0, pred_score) ** 2
            neg_count += 1
    neg_mse = (neg_error / neg_count) ** 0.5
    # assumption: anything above 1 is definitely one/pos class
    pos_errors = [(1.0 - min(1.0, v)) ** 2 for v in pos_scores.values()]
    pos_mse = (sum(pos_errors) / len(pos_errors)) ** 0.5
    err_count = 0
    for i, (src, pos_tgt) in enumerate(pos_exs):
        pos_tgt_score = pos_scores[src]
        if error_tgts[src]:
            err_count += 1
            out.write(f'{i+1:5}\t[FALSE NEG]\t{pos_tgt_score:.4f}\t{src}\t{pos_tgt}\n')
            for err_tgt_score, err_tgt in error_tgts[src]:
                out.write(f'\t[FALSE POS]\t{err_tgt_score:.4f}\t{err_tgt}\n')
        else:
            out.write(f'{i+1:5}\t[TRUE  POS]\t{pos_tgt_score:.4f}\t{src}\t{pos_tgt}\n')
        out.write('\n')
    out.write("\n==================================\n")
    out.write(f'Summary: {err_count} out of {len(pos_exs)} source sentences were scored higher with wrong targets'
              f' ({(100.0 * err_count /len(pos_exs)):.2f}%)\n')
    out.write(f"Positives: {len(pos_exs)}  Negatives: {len(pos_exs)} x {neg_sample_count} = {neg_count}\n")
    out.write(f"Mean-squared diff of positives from 1.0: {pos_mse:.4f}\n")
    out.write(f"Mean-squared diff of negatives from 0.0: {neg_mse:.4f}\n")
    error = (pos_count * pos_mse + neg_count * neg_mse) / (pos_count + neg_count)
    out.write(f"Mean-squared diff (averaged)-----------: {error:.4f}\n")


if __name__ == '__main__':
    import argparse
    import sys
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Source<tab>english sentence per line')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout)
    p.add_argument('-f', '--flags', type=str, help='list of scorers to use. "mcss" to use only the mcss scorer',
                   default='charlen,toklen,copypatn,ascii,mcss')
    p.add_argument('-se', '--src-emb', type=str, help='path to source language embedding (MCSS vectors)')
    p.add_argument('-ee', '--eng-emb', type=str, help='path to english language embedding (MCSS vectors)')
    p.add_argument('-m', '--max-vocab', type=int, help='Max vocabulary size (MCSS vectors)', default=int(1e6))
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
        if 'seed' in args:
            random.seed(args.pop('seed'))
        test(scorer, **args)
    else:
        predict(scorer, **args)
