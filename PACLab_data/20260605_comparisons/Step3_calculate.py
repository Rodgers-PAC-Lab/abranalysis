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
subdf = abra_df.loc['sham',False,:,'Lighthouse_230',:,'LV','R',[73]]
subdf = subdf.reset_index()
orig_x, orig_y, highest_peaks, relevant_troughs = calculate_and_plot_wave_exact(subdf, 1000, 73)
print('Ok I did it!')

