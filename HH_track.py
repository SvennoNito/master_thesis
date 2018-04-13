from HH_cython import hhModel
from HH_helper import *

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import pylab 
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io 

# plot model and error of one objective over generations
def trackOneObjective(feature, gen, pop, feature_list, channels, stim_current, num_gen, dt, fI):
    fitness = []

    pos = feature_list.index(feature)
    for ind in pop:
        fitness.append(ind.fitness.values[pos])
    best_ind = fitness.index(min(fitness))
    best_ind = pop[best_ind]

    params = dict(zip(channels.keys(), best_ind))
    Iext   = buildStimVec(stim_current, 500, 500, dt)
    v, t   = hhModel(dictToListParams(params), Iext, dt, fI)

    if gen == 1:
        global ax1
        fig, ax1 = plt.subplots(2)
        plt.ion()  # enables interactive plotting
        
    # error
    ax1[1].boxplot(fitness, positions=[gen])
    ax1[1].set_xlim(0.5, num_gen - 0.5)
    ax1[1].set_ylim(0, max(max(fitness), 10))
    ax1[1].set_xlabel('generation')
    ax1[1].set_ylabel('error %s' % feature)

    # model
    ax1[0].clear()
    ax1[0].plot(t, v)
    ax1[0].set_xlabel('Time [ms]')
    ax1[0].set_ylabel('Membrane Potential [mV]')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
     
