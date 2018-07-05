import io
import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


def load_vec(emb_path, nmax=1e9):
    vectors = []
    word2id = {}
    logger.info('loading %s' % emb_path)
    with io.open(emb_path, 'r', encoding='utf-8',
                 newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    logger.info("loaded %i pre-trained embeddings." % len(vectors))
    id2word = {v: k for k, v in word2id.items()}
    word_vec = {}
    embeddings = np.vstack(vectors)
    for i in word2id:
        word_vec[i] = embeddings[word2id[i]]
    return embeddings, id2word, word2id, word_vec


def bow(sentences, word_vec, normalize=False):
    """
    Get sentence representations using average bag-of-words.
    """
    embeddings = []
    count = {'tol': 0, 'oov': 0}
    for sent in sentences:
        sentvec = [word_vec[w] for w in sent if w in word_vec]
        for w in sent:
            count['tol'] += 1
            if w not in word_vec:
                count['oov'] += 1
        if normalize:
            sentvec = [v / np.linalg.norm(v) for v in sentvec]
        if len(sentvec) == 0:
            sentvec = [word_vec[list(word_vec.keys())[0]]]
        embeddings.append(np.mean(sentvec, axis=0))
    if count['oov'] / count['tol'] > 0.5:
        logger.debug('# of oov: %s %s' % (count['oov'], count['tol']))
    return np.vstack(embeddings)


def bow_idf(sentences, word_vec, idf_dict=None):
    """
    Get sentence representations using weigthed IDF bag-of-words.
    """
    embeddings = []
    for sent in sentences:
        sent = set(sent)
        list_words = [w for w in sent if w in word_vec and w in idf_dict]
        if len(list_words) > 0:
            sentvec = [word_vec[w] * idf_dict[w] for w in list_words]
            sentvec = sentvec / np.sum([idf_dict[w] for w in list_words])
        else:
            sentvec = [word_vec[list(word_vec.keys())[0]]]
        embeddings.append(np.sum(sentvec, axis=0))
    return np.vstack(embeddings)


class MCSS:

    def __init__(self, src_vec_path, tgt_vec_path, nmax=3e5):
        _, _, _, self.src_vec = load_vec(src_vec_path, nmax=nmax)
        _, _, _, self.tgt_vec = load_vec(tgt_vec_path, nmax=nmax)

    def score(self, src_sent, tgt_sent):
        src_sents = [src_sent.lower().split()]
        tgt_sents = [tgt_sent.lower().split()]
        src_vectors = bow(src_sents, self.src_vec)
        tgt_vectors = bow(tgt_sents, self.tgt_vec)
        return float(cosine_similarity(src_vectors, tgt_vectors))

    def doc_score(self, src_sents: List[str], tgt_sents: List[str]) -> float:
        """Compute the similarity between two documents i.e. two lists of sentences"""
        # FIXME: implement this
        raise Exception('Not Implemented')


if __name__ == '__main__':

    p_src_vec = 'rpi/vectors-si.txt'
    p_tgt_vec = 'rpi/vectors-en.txt'
    mcss = MCSS(p_src_vec, p_tgt_vec)

    s = 'උස අඩි 8 ක් පමණ වන මෙම අලියාගේ වයස අවුරුදු 25 ක් පමණ වන බව සඳහන් .'
    t = 'The 8 feet tall elephant is around 25 - years of age .'
    print(mcss.score(s, t))
    s = 'උස අඩි 8 ක් පමණ වන මෙම අලියාගේ වයස අවුරුදු 25 ක් පමණ වන බව සඳහන් .'
    t = 'this is a test .'
    print(mcss.score(s, t))
