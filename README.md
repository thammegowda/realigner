# Re-Aligner

This code was written during the DARPA LORELEI y3 evaluation week. We found that the LDC data packs had mis aligned sentences. 
This project uses a bunch of heuristics and scoring function to re-align sentences within document.

The hueristics are based on appearance of same numbers and URLs on both sides of bitext.
Scoring functions:
1. MCSS, a multilingual common semantic space based approach which aligns words from both source and target language into same vector space. Then makes alignments based on the similarity of words in this MCSS space.
2. T-table measure. Uses GIZA++ aligner's translation table entries to compute a simple score. The score can be interpreted as probability of source sentence generating target sentence and target sentence generating source sentence as per the given translation table


## How to use:
The `scripts` directory has bunch of scripts (the actual scripts) I used to run. 


