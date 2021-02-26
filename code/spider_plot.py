# -*- coding: utf-8 -*-
"""
Code to plot spider plots of NLP measures.
Author: Dr Sarah E Morgan, 21/02/2021
"""

import numpy as np
import matplotlib.pyplot as plt

# adapted from: https://stackoverflow.com/questions/24659005/radar-chart-with-multiple-scales-on-multiple-axes
class Radar(object):
    def __init__(self, figure, title, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]

        self.n = len(title)
        self.angles = np.arange(0, 360, 360.0/self.n)

        self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=title, fontsize=14)

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.set_rgrids(range(1, 6), angle=angle, labels=label)
            ax.spines['polar'].set_visible(False)
            ax.set_ylim(0, 5)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)


# function to get range from start, end and interval:
def get_range(s1,e1):
    i1=5
    myrange=np.linspace(s1,e1,i1+1)
    myrange=myrange[1:]
    return myrange


# function to transform result to the range desired for spider plot:
def putvec_in_range(result,s_vec,e_vec):
    i_val=5
    result = np.array(result)
    newresult = i_val*(result-s_vec)/(e_vec-s_vec)
    return newresult


# function to shift s_vec: (for visualisation purposes- so that lowest possible value lies on the first gridline, not at the origin)
def shift_svec(s_vec,e_vec):
    s_vec_new = s_vec-(e_vec-s_vec)/4
    return s_vec_new

# function to plot spider plot and save as .png file
def plot_spider(result,s_vec,e_vec,spoke_titles,output_file):
    
    s_vec = shift_svec(s_vec,e_vec)
    fig = plt.figure(figsize=(12, 10))
    lab = [
        np.around(get_range(s_vec[0],e_vec[0]),2),
        np.around(get_range(s_vec[1],e_vec[1]),2),
        np.around(get_range(s_vec[2],e_vec[2]),2),
        np.around(get_range(s_vec[3],e_vec[3]),2),
        np.around(get_range(s_vec[4],e_vec[4]),2),
        np.around(get_range(s_vec[5],e_vec[5]),3),
        np.around(get_range(s_vec[6],e_vec[6]),2)
        ]
    radar = Radar(fig, spoke_titles, lab)
    radar.plot(putvec_in_range(result,s_vec,e_vec),  '-', lw=2, color='b', alpha=0.4, label='Participant')
    radar.ax.legend()

    fig.savefig(output_file)