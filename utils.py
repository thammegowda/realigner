from collections import defaultdict
import random


def scorer_eval(scorer, inp, out, neg_sample_count=20, verbose=True, parse=True, seed=None, **args):
    """
    :param scorer: scorer to evaluate on
    :param inp: input having parallel data (positive examples)
    :param out: output to write results
    :param neg_sample_count: number of negative samples to make
    :param verbose: print out negative
    :param parse: parse the input by splitting it into source and target sentences
    :return: error percentage
    """
    if seed is not None:
        random.seed(seed)
    pos_exs = (line.strip().split('\t') for line in inp) if parse else inp
    pos_exs = list(pos_exs)
    pos_count = len(pos_exs)
    assert pos_count > neg_sample_count
    src_sqs, tgt_seqs = zip(*pos_exs)
    pos_scores = {src: scorer.score(src, tgt) for src, tgt in pos_exs}
    # assert len(pos_scores) == pos_count , 'Unique sentences'
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
            errs_sorted = sorted(error_tgts[src], key=lambda x: x[1], reverse=True)
            errs_sorted = errs_sorted if verbose else errs_sorted[:1]
            for err_tgt_score, err_tgt in errs_sorted:
                out.write(f'\t[FALSE POS]\t{err_tgt_score:.4f}\t{err_tgt}\n')
        else:
            out.write(f'{i+1:5}\t[TRUE  POS]\t{pos_tgt_score:.4f}\t{src}\t{pos_tgt}\n')
        out.write('\n')
    out.write("\n=============SUMMARY=====================\n")
    error_percent = 100.0 * err_count / len(pos_exs)
    out.write(f'Errors: {error_percent:.2f}% \n'
              f'Stats : {err_count} out of {len(pos_exs)} source sentences were scored higher with wrong targets\n')
    out.write(f"Positives: {len(pos_exs)}  Negatives: {len(pos_exs)} x {neg_sample_count} = {neg_count}\n")
    out.write(f"Mean-squared diff of positives from 1.0: {pos_mse:.4f}\n")
    out.write(f"Mean-squared diff of negatives from 0.0: {neg_mse:.4f}\n")
    error = (pos_count * pos_mse + neg_count * neg_mse) / (pos_count + neg_count)
    out.write(f"Mean-squared diff (averaged)-----------: {error:.4f}\n")
    return error_percent
