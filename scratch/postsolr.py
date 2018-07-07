import argparse
import logging as log
from solr import Solr
log.basicConfig(level=log.INFO)
debug_mode = log.getLogger().isEnabledFor(level=log.DEBUG)


def read_docs(lines, fields, overwrite=False):
    assert 'id' in fields
    for line in lines:
        cols = line.strip().split('\t')
        assert len(fields) == len(cols)
        doc = {}
        for field, val in zip(fields, cols):
            doc[field] = val if overwrite or field == 'id' else {'set': val}
        yield doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('url', type=str, help='Solr URL')
    p.add_argument('fields', type=str, help='destination field name such as "textr_t"')
    p.add_argument('-i', '--inp', type=str, help='Input file')
    p.add_argument('--overwrite', action='store_true', help='Set this for posting new documents. '
                                                            'Dont set this for updating docs')
    args = vars(p.parse_args())
    docs = read_docs(args.pop('inp'), args['fields'].split(','), args.pop('overwrite'))
    Solr(args).post_iterator(docs)
