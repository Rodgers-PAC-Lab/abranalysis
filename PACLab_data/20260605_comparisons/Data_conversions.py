import numpy as np
import paclab.abr_misc
import pandas
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os.path

# Import pickles of big_abrs
## Day's Information
datestring = '260603'
day_directory = "_ABR"
experimenter = 'rowan'
metadata_version = "v6"
loudest_dB = 73

## Paths
GUIdata_directory, Pickle_directory = (paclab.abr_misc.get_ABR_data_paths())

# Use daily pickle directory
Pickle_directory = os.path.join(Pickle_directory, datestring, experimenter)
if not os.path.exists(Pickle_directory):
    try:
        os.mkdir(Pickle_directory)
        print("The pickle directory didn't exist. Did you run Step1_loading_aligning.py?")
    except:
        print("No pickle directory exists and this script doesn't have permission to create one.")
        print("Check your Pickle_directory file path.")

# Make 'figures' subdirectory if it doesn't exist yet
if not os.path.exists(os.path.join(Pickle_directory,'figures')):
    try:
        os.mkdir(os.path.join(Pickle_directory,'figures'))
        os.mkdir(os.path.join(Pickle_directory, 'figures', 'svgs'))
        os.mkdir(os.path.join(Pickle_directory, 'figures', 'pngs'))
    except:
        print("No pickle directory exists and this script doesn't have permission to create one.")
        print("Check your Pickle_directory file path.")
if not os.path.exists(os.path.join(Pickle_directory,'figures','svgs')):
    try:
        os.mkdir(os.path.join(Pickle_directory, 'figures', 'svgs'))
    except:
        print("No pickle directory exists and this script doesn't have permission to create one.")
        print("Check your Pickle_directory file path.")
if not os.path.exists(os.path.join(Pickle_directory,'figures','pngs')):
    try:
        os.mkdir(os.path.join(Pickle_directory, 'figures', 'pngs'))
    except:
        print("No pickle directory exists and this script doesn't have permission to create one.")
        print("Check your Pickle_directory file path.")
## Params
sampling_rate = 16000  # TODO: store in recording_metadata

## Load results of Step1
recording_metadata = pandas.read_pickle(os.path.join(Pickle_directory, 'recording_metadata'))

## Load results of Step2
big_triggered_ad = pandas.read_pickle(
    os.path.join(Pickle_directory, 'big_triggered_ad'))
big_triggered_neural = pandas.read_pickle(
    os.path.join(Pickle_directory, 'big_triggered_neural'))

## Load results of Step3
big_abrs = pandas.read_pickle(
    os.path.join(Pickle_directory, 'big_abrs'))

# Rename 'label' as 'level(dB)
ABRA_big_abrs = big_abrs.copy()
ABRA_big_abrs = ABRA_big_abrs.rename_axis(index={"label":"Level(dB)"})

# It requires a frequency parameter even though that's not relevant for clicks
ABRA_big_abrs['Freq(Hz)'] = 1000

