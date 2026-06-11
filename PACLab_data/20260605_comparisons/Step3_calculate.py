import torch
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('TkAgg')
# from utils.models import default_peak_finding_model, default_thresholding_model
# import streamlit as st
# from utils.ui import *
# from utils.processFiles import db_value
import os
import datetime
import json
import numpy as np
import pandas
import ABR2025
import my.plot
import matplotlib.pyplot as plt
import tqdm
from utils.models import CNN
from utils.calculate_PACLAB import interpolate_and_smooth, peak_finding, default_peak_finding_model, calculate_and_plot_wave_exact

## Plotting defaults
my.plot.manuscript_defaults()
my.plot.font_embed()
MU = chr(956)
## Paths
# Load the required file filepaths.json (see README)
with open('filepaths.json') as fi:
    paths = json.load(fi)

# Parse into paths to raw data and output directory
raw_data_directory = paths['raw_data_directory']
output_directory = paths['output_directory']

## Params
# Outlier params
abs_max_sigma = 3
stdev_sigma = 3

## Load metadata
mouse_metadata = pandas.read_hdf(
    os.path.join(output_directory,'metadata.hd5'),key="mouse_metadata")
experiment_metadata = pandas.read_hdf(
    os.path.join(output_directory,'metadata.hd5'),key="experiment_metadata")
recording_metadata = pandas.read_hdf(
    os.path.join(output_directory,'metadata.hd5'),key="recording_metadata")

## Load previous results
# Load results of Step2_avging
big_abrs = pandas.read_hdf(
    os.path.join(output_directory,'abr_avgs.hd5'),key='big_abrs')

# Loudest dB
loudest_db = big_abrs.index.get_level_values('label').max()

# Remove the 40 pre-click samples
abra_df = big_abrs.loc[:, 0:].copy()
abra_df = abra_df.rename_axis(index={"label":"Level(dB)"})
abra_df.insert(0,'Freq(Hz)',1000)
abra_df = abra_df.set_index('Freq(Hz)', append=True)

# This is a one-off proof of concept to make sure the calculations work here
subdf = abra_df.loc['sham',False,:,'Lighthouse_230',:,'LV','R',[73]]
# subdf = subdf.reset_index()
orig_x, orig_y, highest_peaks, relevant_troughs = calculate_and_plot_wave_exact(-subdf, 1000, 73)
print('Ok I did it!')

# # Test the plotting one-off
# plt.plot(orig_x, -subdf.loc[:,0:].mean())
# plt.plot(orig_x[highest_peaks][0], orig_y[highest_peaks][0], '.', color='red')
# plt.plot(orig_x[relevant_troughs][0], orig_y[relevant_troughs][0], '.', color='darkred')
# plt.plot(orig_x[highest_peaks][1], orig_y[highest_peaks][1], '.', color='blue')
# plt.plot(orig_x[relevant_troughs][1], orig_y[relevant_troughs][1], '.', color='darkblue')
# plt.plot(orig_x[highest_peaks][2], orig_y[highest_peaks][2], '.', color='limegreen')
# plt.plot(orig_x[relevant_troughs][2], orig_y[relevant_troughs][2], '.', color='darkgreen')
# plt.plot(orig_x[highest_peaks][3], orig_y[highest_peaks][3], '.', color='gold')
# plt.plot(orig_x[relevant_troughs][3], orig_y[relevant_troughs][3], '.', color='darkgoldenrod')
# plt.plot(orig_x[highest_peaks][4], orig_y[highest_peaks][4], '.', color='hotpink')
# plt.plot(orig_x[relevant_troughs][4], orig_y[relevant_troughs][4], '.', color='darkmagenta')

pre_HL = abra_df.xs(False,level='after_HL')

# HL_type is irrelevant if we're only looking at pre_HL
pre_HL = pre_HL.droplevel('HL_type',axis='index')
pre_HL = pre_HL.groupby(['mouse','channel','speaker_side', 'Level(dB)', 'Freq(Hz)']).mean()
gobj = pre_HL.groupby(['mouse','channel','speaker_side'])
for (mouse, channel ,speaker_side),subdf in gobj:
    # print('(hacker voice) Im in')
    # Make a plot
    f, ax = plt.subplots(figsize=(10,6))

    # Make the colorbar
    aut_colorbar = my.plot.generate_colorbar(
        len(subdf.index), mapname='inferno', start=0, stop=0.85)

    color_df = pandas.DataFrame(aut_colorbar,
        index=subdf.index.sort_values(ascending=True))

    # Iterate over sound levels
    sound_levels = subdf.index.get_level_values('Level(dB)').unique()
    sound_levels = sound_levels.sort_values(ascending=False)
    all_xcoord_l = []
    all_ycoord_l = []
    for i_db in sound_levels:
        # Get peaks and troughs
        topl = subdf.reset_index()
        topl = topl.loc[topl['Level(dB)']==i_db]
        topl = topl.set_index(subdf.index.names)
        orig_x, orig_y, highest_peaks, relevant_troughs = calculate_and_plot_wave_exact(-topl, 1000, i_db)


        ax.plot(orig_x, -topl.loc[:, 0:].mean(),
            color=color_df.xs(i_db, level='Level(dB)').values, lw=.75, label=i_db)
        # ax.plot(orig_x[highest_peaks][0], orig_y[highest_peaks][0], '.', color='red')
        # ax.plot(orig_x[relevant_troughs][0], orig_y[relevant_troughs][0], '.', color='darkred')
        # ax.plot(orig_x[highest_peaks][1], orig_y[highest_peaks][1], '.', color='blue')
        # ax.plot(orig_x[relevant_troughs][1], orig_y[relevant_troughs][1], '.', color='darkblue')
        # ax.plot(orig_x[highest_peaks][2], orig_y[highest_peaks][2], '.', color='limegreen')
        # ax.plot(orig_x[relevant_troughs][2], orig_y[relevant_troughs][2], '.', color='darkgreen')
        # ax.plot(orig_x[highest_peaks][3], orig_y[highest_peaks][3], '.', color='gold')
        # ax.plot(orig_x[relevant_troughs][3], orig_y[relevant_troughs][3], '.', color='darkgoldenrod')
        # ax.plot(orig_x[highest_peaks][4], orig_y[highest_peaks][4], '.', color='hotpink')
        # ax.plot(orig_x[relevant_troughs][4], orig_y[relevant_troughs][4], '.', color='darkmagenta')

        colors_l = ['red', 'darkred', 'blue', 'darkblue', 'green', 'limegreen',
        'gold', 'darkgoldenrod', 'hotpink', 'darkmagenta']

        d2 = {
            'wave1_pk': [orig_x[highest_peaks][0], orig_y[highest_peaks][0]],
            'wave1_tr': [orig_x[relevant_troughs][0], orig_y[relevant_troughs][0]],
            'wave2_pk': [orig_x[highest_peaks][1], orig_y[highest_peaks][1]],
            'wave2_tr': [orig_x[relevant_troughs][1], orig_y[relevant_troughs][1]],
            'wave3_pk': [orig_x[highest_peaks][2], orig_y[highest_peaks][2]],
            'wave3_tr': [orig_x[relevant_troughs][2], orig_y[relevant_troughs][2]],
            'wave4_pk': [orig_x[highest_peaks][3], orig_y[highest_peaks][3]],
            'wave4_tr': [orig_x[relevant_troughs][3], orig_y[relevant_troughs][3]],
            'wave5_pk': [orig_x[highest_peaks][4], orig_y[highest_peaks][4]],
            'wave5_tr': [orig_x[relevant_troughs][4], orig_y[relevant_troughs][4]],
        }
        savename = 'ABRA_pks_' + mouse + '_' + channel + '_' + speaker_side
        for i_pk, i_color in zip(d2, colors_l):
            try:
                ax.plot(d2[i_pk][0], d2[i_pk][1], '.', color=i_color)
            except:
                print('Failed on '+ mouse + '_' + channel + '_' +
                      speaker_side + ' ' + i_db + ' dB, peak ' + str(i_pk))
    f.suptitle(mouse + ' ' + channel + ' ' + speaker_side)

    f.savefig(os.path.join(
        output_directory, 'figures', savename+'.png'), dpi=300)
    plt.close(f)