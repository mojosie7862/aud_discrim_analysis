import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import matplotlib as mpl

#bring in tracking data, remove 0,0 points
def import_exp_tracking(file):
    trans_xy_df = pd.read_csv(file, index_col=0, dtype={'frame_id': int, 'frame': int, 'frame_time_split': float,
                                                                      'time': float, 'run_id': str, 'run_num': str, 'paradigm': str,
                                                                      'time_var': str, 'HeadX': int, 'HeadY': int,
                                                                      'TailBaseX': int, 'TailBaseY': int, 'Heading': float})
    og_len = len(trans_xy_df)
    trans_xy_df = trans_xy_df[trans_xy_df['HeadX']+trans_xy_df['HeadY'] > 0]
    filtered_len = len(trans_xy_df)
    filtered_rows = og_len-filtered_len

    print("removed", filtered_rows, "rows in", file)
    return trans_xy_df
sscf_av_tracking_df = import_exp_tracking('tracking_zf_data_10sscf_av.csv')
ssfm_av_tracking_df = import_exp_tracking('tracking_zf_data_10ssfm_av.csv')
ssfm_rew_tracking_df = import_exp_tracking('tracking_zf_data_10ssfm_rew.csv')
sscf_rew_tracking_df = import_exp_tracking('tracking_zf_data_11sscf_rew.csv')

print(len(sscf_av_tracking_df))
print(len(ssfm_av_tracking_df))
print(len(ssfm_rew_tracking_df))
print(len(sscf_rew_tracking_df))

av_dfs = [sscf_av_tracking_df, ssfm_av_tracking_df]
rew_dfs = [ssfm_rew_tracking_df, sscf_rew_tracking_df]
av_tracking_df = pd.concat(av_dfs)
rew_tracking_df = pd.concat(rew_dfs)


#paradigm dataframes
run_nums = ['1', '2', '3', '4', '5', '6']

cf_runs_df = av_tracking_df[av_tracking_df['paradigm'] == 'cf']
fm_runs_df = av_tracking_df[av_tracking_df['paradigm'] == 'fm']
b1_runs_df = av_tracking_df[av_tracking_df['paradigm'] == 'b1']
b2_runs_df = av_tracking_df[av_tracking_df['paradigm'] == 'b2']

cf_ids = set(cf_runs_df['zf_id'])
cf_runs = sorted(list(set(cf_runs_df['run_id'])))
fm_ids = set(fm_runs_df['zf_id'])
fm_runs = sorted(list(set(fm_runs_df['run_id'])))

#dfs with all cf and fm baseline data
cfb1_baseline_df = b1_runs_df[b1_runs_df['zf_id'].isin(list(cf_ids))]
cfb2_baseline_df = b2_runs_df[b2_runs_df['zf_id'].isin(list(cf_ids))]
fmb1_baseline_df = b1_runs_df[b1_runs_df['zf_id'].isin(list(fm_ids))]
fmb2_baseline_df = b2_runs_df[b2_runs_df['zf_id'].isin(list(fm_ids))]

#lists of baseline run ids by cf/fm and b1/b2
cf_b1s = list(set(cfb1_baseline_df['run_id']))
cf_b2s = list(set(cfb2_baseline_df['run_id']))
fm_b1s = list(set(fmb1_baseline_df['run_id']))
fm_b2s = list(set(fmb2_baseline_df['run_id']))

#initiating a few variables
cfrun_dfs = {}
fmrun_dfs = {}
cfb1_dfs = {}
cfb2_dfs = {}
cf_int_ppi_dict = {}
cf_run_ppi_dict = {}
cfb1_ppi = []
cfb2_ppi = []
fm_int_ppi_dict = {}
fm_run_ppi_dict = {}
fmb1_ppi = []
fmb2_ppi = []


#calculate ppi over timecouse
def calc_ppi(run_data):
    screenside_df = run_data[run_data['HeadX'] > 315] #410
    farside_df = run_data[run_data['HeadX'] < 315] #210
    pref_time = sum(screenside_df['frame_time_split'])
    nonpref_time = sum(farside_df['frame_time_split'])
    # if pref_time+nonpref_time == 0:
    ppi = pref_time / (pref_time + nonpref_time)
    return ppi