## Remove outliers and aggregates ABRs over trials
# Writes out:
#   big_abrs - averaged ABRs
#   trial_counts - trial counts
#   averaged_abrs_by_date - averaged ABRs aggregated by date
#   averaged_abrs_by_mouse - averaged ABRs aggregated by mouse
#
# Plots:
#   PLOT_SINGLE_TRIAL_ABR
#   PLOT_TRIAL_AVERAGED_ABR
#   PLOT_POSITIVE_AND_NEGATIVE_CLICKS


import os
import datetime
import json
import numpy as np
import pandas
import ABR2025
import my.plot
import matplotlib.pyplot as plt
import tqdm


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
# Load results of Step1_PACLab_loading
big_triggered_neural = pandas.read_hdf(
    os.path.join(output_directory,'big_data.hd5'),key='big_triggered_neural')
big_click_params = pandas.read_hdf(
    os.path.join(output_directory,'big_data.hd5'),key='big_click_params')

# Loudest dB
loudest_db = big_triggered_neural.index.get_level_values('label').max()

## Join on speaker_side
# Should have done this at the same time as joining channel
idx = big_triggered_neural.index.to_frame().reset_index(drop=True)
idx = idx.join(
    recording_metadata['speaker_side'], on=['date', 'mouse', 'recording'])
big_triggered_neural.index = pandas.MultiIndex.from_frame(idx)

# Reorder levels
big_triggered_neural = big_triggered_neural.reorder_levels([
    'date', 'mouse', 'recording', 'channel', 'speaker_side', 'label',
    'polarity', 't_samples']).sort_index()

## Drop outlier trials, separately by channel
# Consider outliers separately for every channel on every recording
group_by = ['date', 'mouse', 'recording', 'channel', 'speaker_side']
gobj = big_triggered_neural.groupby(group_by)


# We use this helper function to drop the groupby levels, otherwise
# the levels get duplicated by gobj.apply
def drop_outliers(df):
    res = ABR2025.signal_processing.trim_outliers(
        df.droplevel(group_by),
        abs_max_sigma=abs_max_sigma,
        stdev_sigma=stdev_sigma,
    )

    return res


# Apply the drop
big_triggered_neural2 = gobj.apply(drop_outliers)

# Reorder levels to be like triggered_neural
big_triggered_neural2 = big_triggered_neural2.reorder_levels(
    big_triggered_neural.index.names).sort_index()

## Count the number of trials remaining in each recording
trial_counts = big_triggered_neural2.groupby(
    ['date', 'mouse', 'speaker_side', 'recording', 'label', 'channel']).size()

## Aggregate
# Average out the trial
by_polarity = big_triggered_neural2.groupby(
    [lev for lev in big_triggered_neural2.index.names if lev != 't_samples']
).mean()

# Compute the big_abrs by adding over polarity
big_abrs = 0.5 * (
        by_polarity.xs(True, level='polarity') +
        by_polarity.xs(False, level='polarity')
)

# Compute the big_arts by subtracting over polarity
big_arts = 0.5 * (
        by_polarity.xs(True, level='polarity') -
        by_polarity.xs(False, level='polarity')
)

## Join metadata on big_abrs
# Join after_HL and n_experiment onto big_abrs
big_abrs = my.misc.join_level_onto_index(
    big_abrs,
    experiment_metadata.set_index(['mouse', 'date'])[['after_HL', 'n_experiment']],
    join_on=['mouse', 'date']
    )

# Drop the now unnecessary level 'date' (replaced with n_experiment)
big_abrs = big_abrs.droplevel('date')

# Join HL_type onto big_abrs
big_abrs = my.misc.join_level_onto_index(
    big_abrs,
    mouse_metadata.set_index('mouse')['HL_type'],
    join_on='mouse',
    )


## Join metadata on trial_counts in the same way
# Join after_HL and n_experiment onto trial_counts
trial_counts = my.misc.join_level_onto_index(
    trial_counts,
    experiment_metadata.set_index(['mouse', 'date'])[['after_HL', 'n_experiment']],
    join_on=['mouse', 'date']
    )

# Drop the now unnecessary level 'date' (replaced with n_experiment)
trial_counts = trial_counts.droplevel('date')

# Join HL_type onto big_abrs
trial_counts = my.misc.join_level_onto_index(
    trial_counts,
    mouse_metadata.set_index('mouse')['HL_type'],
    join_on='mouse',
    )

# Reorder level to match big_abrs
trial_counts = trial_counts.reorder_levels(big_abrs.index.names).sort_index()


## Further aggregate big_abrs
# Mean out recording, leaving n_experiment
averaged_abrs_by_date = big_abrs.groupby(
    [lev for lev in big_abrs.index.names if lev != 'recording']
    ).mean()

# Keep only the first experiment from each mouse
# Better to keep the first than to average, because the two experiments
# may be different-looking or out of phase
averaged_abrs_by_mouse = averaged_abrs_by_date.xs(0, level='n_experiment')


## Dump overall N
# Estimate duration as the last click time
recording_duration = big_click_params.groupby(
    ['date', 'mouse', 'recording']).apply(lambda df: df.index[-1][-1]
    ) / 16e3
n_recordings = len(recording_duration)
n_experiments = len(recording_duration.groupby(['date', 'mouse']).sum())
n_mice = len(recording_duration.groupby('mouse').sum())
quantiles = recording_duration.quantile((0, .25, .5, .75, 1))

# Get range of surgical experiment dates
surgery_df = experiment_metadata.copy()
surgery_dates = (surgery_df['date'] - surgery_df['HL_date']).dropna().sort_values()
surgery_dates = surgery_dates.value_counts().sort_index()

# Print N
stats_filename = 'figures/STATS__N_OVERALL'
with open(stats_filename, 'w') as fi:
    fi.write(stats_filename + '\n')
    fi.write(f'n = {n_mice} mice; {n_recordings} recordings; {n_experiments} experiments\n')
    fi.write(f'duration quantiles:\n{str(quantiles)}\n')
    fi.write(f'surgery date range:\n{str(surgery_dates)}')

# Echo
with open(stats_filename) as fi:
    print(''.join(fi.readlines()))

## Store
big_abrs.to_hdf(os.path.join(output_directory,'abr_avgs.hd5'), key='big_abrs')
averaged_abrs_by_date.to_hdf(os.path.join(output_directory,'abr_avgs.hd5'),
    key='averaged_abrs_by_date')
averaged_abrs_by_date.to_pickle(os.path.join(output_directory, 'averaged_abrs_by_date'))
averaged_abrs_by_mouse.to_hdf(os.path.join(output_directory,'abr_avgs.hd5'),
    key='averaged_abrs_by_mouse')
averaged_abrs_by_mouse.to_pickle(os.path.join(output_directory, 'averaged_abrs_by_mouse'))
trial_counts.to_hdf(os.path.join(output_directory,'abr_avgs.hd5'),
    key='trial_counts')
trial_counts.to_pickle(os.path.join(output_directory, 'trial_counts'))