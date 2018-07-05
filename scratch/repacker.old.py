
import argparse
import sys
import logging as log
from lxml import etree
from typing import Dict, TextIO, BinaryIO, Union
import gzip

log.basicConfig(level=log.INFO)
debug_mode = log.getLogger().isEnabledFor(level=log.DEBUG)

def read_alignment(inp: TextIO) -> Dict[str, str]:
    """Reads alignment file into a dictionary"""
    recs = (line.strip().split('\t')[:2] for line in inp)
    recs = (tuple(col.strip() for col in row) for row in recs)
    return {k: v for k, v in recs}


def re_align(elisa_xml: Union[TextIO, BinaryIO],
             out: TextIO, mapping: Dict[str, str],
             src_field: str='ULF_LRLP_TOKENIZED_SOURCE',
             tgt_field: str='ULF_LRLP_TOKENIZED_TARGET'):
    src2tgt = mapping
    tgt2src = {t: s for s, t in src2tgt.items()}
    assert len(src2tgt) == len(tgt2src)

    if elisa_xml.name.endswith('.gz'):
        elisa_xml.close()
        elisa_xml = gzip.open(elisa_xml.name)

    segments = etree.iterparse(elisa_xml, events=('end',), tag='SEGMENT')
    count, skip_count = 0, 0

    tgt_val_buffer = {}   # pending targets
    src_val_buffer = {}   # pending sources
    for _, seg_el in segments:
        src_id = seg_el.xpath('.//SOURCE/@id')[0]
        tgt_id = seg_el.xpath('.//TARGET/@id')[0]
        src_val = seg_el.xpath(f'.//{src_field}/text()')[0]
        tgt_val = seg_el.xpath(f'.//{tgt_field}/text()')[0]
        seg_el.clear()

        skip = False
        if src_id in src2tgt:
            mapped_tgt_id = src2tgt[src_id]
            if mapped_tgt_id == tgt_id:       # matching is not changed
                assert tgt2src[tgt_id] == src_id
                del src2tgt[src_id], tgt2src[tgt_id]
            elif mapped_tgt_id in tgt_val_buffer:  # target side needs to be changed
                log.info(f" {src_id} --> {mapped_tgt_id} (old: {tgt_id}) ")
                tgt_val_buffer[tgt_id] = tgt_val
                tgt_id, tgt_val = mapped_tgt_id, tgt_val_buffer[mapped_tgt_id]
            else:
                log.debug(f"Skip : {tgt_id}")
                skip = True     # this tgt_id was never mapped
        elif tgt_id in tgt2src:
            mapped_src_id = tgt2src[tgt_id]
            if mapped_src_id in src_val_buffer:  # source side needs to be changed
                log.info(f"  (old: {src_id}) {src_id} --> {mapped_tgt_id} (old: {tgt_id}) ")
                src_val_buffer[src_id] = src_val
                src_id, src_val = mapped_src_id, src_val_buffer[mapped_src_id]
            else:
                log.debug(f"Skip : {src_id}")
                skip = True  # this src_id was never mapped

        if skip:
            skip_count += 1
            continue

        out.write(f'{src_id}\t{tgt_id}\t{src_val}\t{tgt_val}\n')
        count += 1
        if count > 1000000:
            break
    log.info(f"Wrote {count} record to output. Skipped {skip_count} records")
    elisa_xml.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-e', '--elisa-xml', type=argparse.FileType('r'), default=sys.stdin,
                   help='elisa xml file')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file')
    p.add_argument('-a', '--alignment', type=argparse.FileType('r'), required=True,
                   help="Path to alignment file. Format=SRC_ID<tab>TGT_ID")
    p.add_argument('-s', '--src-field', type=str, default='ULF_LRLP_TOKENIZED_SOURCE', help='Source Field')
    p.add_argument('-t', '--tgt-field', type=str, default='ULF_LRLP_TOKENIZED_TARGET', help='Target Field')

    args = vars(p.parse_args())
    matching = read_alignment(args.pop('alignment'))
    re_align(mapping=matching, **args)
