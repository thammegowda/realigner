import logging as log

import xml.etree.ElementTree as et

import argparse
import sys
import glob

log.basicConfig(level=log.INFO)
debug_mode = log.getLogger().isEnabledFor(level=log.DEBUG)


def read_doc_id_mapping(aln_file):
    aln_el = et.parse(aln_file).getroot()
    return aln_el.attrib["source_id"], aln_el.attrib["translation_id"]


def read_doc_alignments(aln_dir):
    aln_files = glob.glob(f'{aln_dir}/*.aln.xml')
    yield from (read_doc_id_mapping(f) for f in aln_files)


def write_out(recs, out):
    for rec in recs:
        out.write('\t'.join(rec))
        out.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-d', '--dir', type=str, help='Input directory having .aln.xml files', required=True)
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout, help='Output file path')
    args = vars(p.parse_args())
    recs = read_doc_alignments(args['dir'])
    write_out(recs, args['out'])
