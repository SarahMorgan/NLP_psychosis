# -*- coding: utf-8 -*-
"""
Code to calculate NLP measures from a speech excerpt, as in Morgan et al 2021:
https://doi.org/10.1101/2021.01.04.20248717
Calculates NLP measures from basic_meas, coh_meas and tangent_meas and also plots the results as a spider plot.
Please cite the paper above if you use this code for your own work.
Author: Dr Sarah E Morgan, 21/02/2021
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# load in functions required:
from basic_meas import *
from coh_meas import *
from tangent_meas import *
from spider_plot import *

# import file that contains participant's speech excerpt:
speechfilename = 'c:\\Users\\sem91\\Documents\\Research\\speech\\github_code\\speech_example1.txt'
speechfile = open(speechfilename, 'rt')
speechtext = speechfile.read()
speechfile.close()

# import file that contains 'ground truth' picture description:
picfilename = 'c:\\Users\\sem91\\Documents\\Research\\speech\\TAT\\picture7.txt'
picfile = open(picfilename, 'rt')
pictext = picfile.read()
picfile.close()

# Start by removing any text inside brackets: ([...] etc)
speechtext = remove_text_inside_brackets(speechtext)

# calculate NLP measures:
basic_all=meas_basic(speechtext)
coh_all=meas_coh(speechtext)
tangent_all=meas_tangent(pictext,speechtext)

# concatenate:
result = basic_all + coh_all + tangent_all

######### Plot spider plot:

# set spoke titles:
spoke_titles = ['No. words', 'No. sent.','Sent. length','Coh.','Max similarity','Tangent','On-topic']

# set start and end points for axes:
# (values here are min and max values from descriptions of picture 7 from all subjects in https://www.medrxiv.org/content/10.1101/2021.01.04.20248717v1)
s_vec=np.array([36,4,3.6,0.2337,0.5482,-0.0758,0.2929])
e_vec=np.array([194,21,38.5,0.6824,1,0.0294,0.5424])

# plot spider plot:
output_file='spider_result.png' # filename for output spider plot
plot_spider(result,s_vec,e_vec,spoke_titles,output_file)