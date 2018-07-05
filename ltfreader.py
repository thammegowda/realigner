# Author :  Thamme Gowda ;; Created : July 04, 2018
import logging as log
import lxml.etree as et
from collections import OrderedDict
import argparse
import sys
import glob

log.basicConfig(level=log.INFO)
debug_mode = log.getLogger().isEnabledFor(level=log.DEBUG)


class Doc:
    def __init__(self, doc_id, lang):
        self.doc_id = doc_id
        self.lang = lang
        self.segs = OrderedDict()

    def add_seg(self, seg_id, text):
        self.segs[seg_id] = text

    def get_segs(self):
        return self.segs.items()

    def get_seg(self, seg_id):
        return self.segs[seg_id]

    def to_recs(self):
        return [(self.doc_id, seg_id, text) for seg_id, text in self.get_segs()]


def read_ltf_docs(path):
    with open(path):
        root = et.parse(path)
        for doc_el in root.xpath('.//DOC'):
            doc = Doc(doc_id=doc_el.attrib['id'], lang=doc_el.attrib['lang'])
            for seg_el in doc_el.xpath('.//SEG'):
                seg_id = seg_el.attrib['id']
                tok_text = " ".join(seg_el.xpath('.//TOKEN/text()'))
                doc.add_seg(seg_id, tok_text)
            yield doc


def read_ltf_doc(path):
    docs = list(read_ltf_docs(path))
    assert len(docs) == 1
    return docs[0]


def read_ltf_dir(dir_path):
    paths = glob.glob(f'{dir_path}/*.ltf.xml')
    log.info(f"Found {len(paths)} files")
    for path in paths:
        yield from read_ltf_docs(path)


def write_out(docs, out):
    for doc in docs:
        for rec in doc.to_recs():
            out.write('\t'.join(rec))
            out.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-d', '--dir', type=str, help='Input directory having LTF files')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout, help='Output file path')
    args = vars(p.parse_args())
    docs = read_ltf_dir(args['dir'])
    write_out(docs, args['out'])
