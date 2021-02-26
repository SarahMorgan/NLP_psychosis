# NLP_psychosis


## Usage:

This repository contains code to calculate the NLP measures employed in Morgan et al 2021: https://doi.org/10.1101/2021.01.04.20248717.
If you use this code in your own research, please cite the paper above.

The code is written for Python 3 and can be found in the 'code' folder.

Briefly, the scripts have the following functions:

- 'get_meas.py' is the main script, which calls the other scripts listed below to calculate the NLP measures and plot the results as a spider plot.
- 'basic_meas.py' calculates basic NLP measures (number of words, number of sentences and mean sentence length).
- 'coh_meas.py' calculates semantic coherence and the maximum similarity (repetition) measure.
- 'tangent_meas.py' calculates tangentiality and on-topic score.
- 'spider_plot.py' contains scripts to plot a spider plot, to visualise the results.

Example speech excerpts and corresponding spider plots are given in the 'examples' folder.

## Notes:

Both 'coh_meas.py' and 'tangent_meas.py' rely on having access to a pre-downloaded word2vec model. I used a slim version of the pre-trained Google News word2vec model, which can be downloaded here: https://github.com/eyaler/word2vec-slim. You will need to update 'coh_meas.py' and 'tangent_meas.py' with the correct path to this model on your computer.

To calculate tangentiality and the on-topic score, you need a 'ground truth' text to compare the speech excerpt to (see Morgan et al 2021 for full details). In Morgan et al 2021, since our speech excerpts were descriptions of pictures from the Thematic Apperception Test (Murray et al, 1943), we used ground-truth descriptions of the TAT pictures, taken from: https://www.psychestudy.com/general/personality/detailed-procedure-thematic-procedure-test. These descriptions are provided in the folder 'examples/TAT' for reference. If using different stimuli (e.g. different pictures), you could substitute these descriptions for something else. However, please note that for the code to work correctly, the ground truth description text must be exactly 1 sentence long (not longer), and contain at least one word present in the word2vec model.

This code is designed for use with speech excerpts in English, and would need to be adapted before use with other languages.

Note that in the paper above we also calcualted speech graph measures and ambiguous pronoun code, for which code by other authors is already openly available here: http://neuro.ufrn.br/softwares/speechgraphs ([Mota et al 2012](https://doi.org/10.1371/journal.pone.0034928) and [Mota et al 2014](https://doi.org/10.1038/srep03691)) and here: https://github.com/kentonl/e2e-coref ([Lee et al 2017](https://arxiv.org/abs/1707.07045)).
