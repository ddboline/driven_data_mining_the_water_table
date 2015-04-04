#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module to plot data, create simple html page

@author: Daniel Boline <ddboline@gmail.com>
"""

import os

import matplotlib
matplotlib.use('Agg')
import pylab as pl

import numpy as np
#from pandas.tools.plotting import scatter_matrix

def create_html_page_of_plots(list_of_plots, prefix='html'):
    """
    create html page with png files
    """
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    os.system('mv *.png %s' % prefix)
    #print(list_of_plots)
    idx = 0
    htmlfile = open('%s/index_0.html' % prefix, 'w')
    htmlfile.write('<!DOCTYPE html><html><body><div>\n')
    for plot in list_of_plots:
        if idx > 0 and idx % 200 == 0:
            htmlfile.write('</div></html></html>\n')
            htmlfile.close()
            htmlfile = open('%s/index_%d.html' % (prefix, (idx//200)), 'w')
            htmlfile.write('<!DOCTYPE html><html><body><div>\n')
        htmlfile.write('<p><img src="%s"></p>\n' % plot)
        idx += 1
    htmlfile.write('</div></html></html>\n')
    htmlfile.close()

def plot_data(indf, prefix='html'):
    """
    create scatter matrix plot, histograms
    """
    list_of_plots = []
#    scatter_matrix(indf)
#    pl.savefig('scatter_matrix.png')
#    list_of_plots.append('scatter_matrix.png')

    for col in indf:
        pl.clf()
#        v = indf[col]
#        nent = len(v)
#        hmin, hmax = v.min(), v.max()
#        xbins = np.linspace(hmin,hmax,nent)
#        hmin, hmax, nbin = BOUNDS[col]
#        xbins = np.linspace(hmin, hmax, nbin)
        print col
        indf[col].hist(histtype='step', normed=True)
        pl.title(col)
        pl.savefig('%s_hist.png' % col)
        list_of_plots.append('%s_hist.png' % col)

    create_html_page_of_plots(list_of_plots, prefix)
    return
