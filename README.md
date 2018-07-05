## Sentence (re) aligner based on MCSS

Crated to realign LDC LORELEI packages (incident languages with bad alignment)


#### How to use :
  - see  `scripts/realign-mcss.sh`
  - see  `scripts/realign-tg.sh`

```
usage: realigner.py [-h] -fd FOUND_DIR -l SRC_LANG [-o OUT_DIR] [-se SRC_EMB]
                    [-ee ENG_EMB] [-th THRESHOLD] [-nt THREADS] [-f FLAGS]
                    [-d]

optional arguments:
  -h, --help            show this help message and exit
  -fd FOUND_DIR, --found-dir FOUND_DIR
                        Path to "found" dir that has eng and xyz lan (default:
                        None)
  -l SRC_LANG, --lang SRC_LANG
                        source language code (default: None)
  -o OUT_DIR, --out-dir OUT_DIR
  -se SRC_EMB, --src-emb SRC_EMB
                        path to source language embedding (MCSS vectors)
                        (default: None)
  -ee ENG_EMB, --eng-emb ENG_EMB
                        path to english language embedding (MCSS vectors)
                        (default: None)
  -th THRESHOLD, --threshold THRESHOLD
                        threshold score below which the sentence pairs must be
                        ignored (default: 0.0)
  -nt THREADS, --threads THREADS
                        Number of threads to use (default: 2)
  -f FLAGS, --flags FLAGS
                        comma separated list of scorers to use. For example
                        set -f "mcss" to use only MCSS or "copypatn,mcss" to
                        use copy pattern scorer and MCSS (default:
                        charlen,toklen,copypatn,ascii,mcss)
  -d, --debug           Turn on the debug mode (default: False)
  ```