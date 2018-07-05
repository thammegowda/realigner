# Author :  Thamme Gowda ;; Created : July 04, 2018
"""
A scorer for mining parallel/comparable sentences from comparable documents
"""
import re
import logging as log
log.basicConfig(level=log.INFO)


class UnifiedScorer:

    must_reject = -20
    must_accept = +20
    may_reject = -1
    may_accept = +1
    not_sure = 0

    copy_patterns = [re.compile(p) for p in [r'(\d+)', r'(https?://[^ ]+)']]

    def __init__(self, mcss=None, flags='charlen,toklen,copypatn,ascii,mcss', debug=False):
        self.mcss = mcss
        flags = flags.split(',') if type(flags) is str else flags
        mapping = {
            'copypatn': self.copy_score,
            'toklen': self.toklen_score,
            'charlen': self.charlen_score,
            'ascii': self.ascii_ratio_score,
            'mcss': self.mcss_score,
        }
        flags = flags.split(',') if type(flags) is str else flags
        self.scorers = [mapping[flag] for flag in flags]
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
            return self.may_reject

    def toklen_score(self, src: str, tgt: str) -> float:
        ratio = (1 + len(src.split())) / (1 + len(tgt.split()))
        if 0.33 <= ratio <= 3.0:
            return self.not_sure
        else:
            return self.may_reject

    def mcss_score(self, src: str, tgt: str) -> float:
        score = self.mcss.score(src, tgt)
        # scale it to make it in similar range as other scores
        scaled_score = score * 2
        if self.debug:
            log.debug(f'MCSS score {score}, scaled : {scaled_score}: {src} {tgt}')
        return scaled_score

    def ascii_ratio_score(self, src: str, tgt: str) -> float:
        """Check that there are approximately same ratio of punctuations and numerals (2x). exclude alphabets"""
        src_ct = 1.0 + sum(1 for c in src if ord(c) < 256 and not c.isalpha())
        tgt_ct = 1.0 + sum(1 for c in tgt if ord(c) < 256 and not c.isalpha())
        if 0.5 <= src_ct / tgt_ct <= 2.0:
            return self.may_accept
        else:
            return self.may_reject

    def score(self, src: str, tgt: str) -> float:
        # negative means No, positive means yes
        tot_score = 0.0
        for scorer in self.scorers:
            tot_score += scorer(src, tgt)
            if tot_score >= self.must_accept or tot_score <= self.must_reject:
                break  # abort the scoring here
        if self.debug:
            log.debug(f'score:{tot_score} :: SRC: {src} \t\t TGT: {tgt}')
        return tot_score
