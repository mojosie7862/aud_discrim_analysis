import pandas as pd
from statistics import stdev
import matplotlib.pyplot as plt
import numpy as np

#bring in tracking data, remove 0,0 points
trans_xy_df = pd.read_csv('tracking_zf_data.csv', index_col=0, dtype={'frame_id': int, 'frame': int, 'frame_time_split': float,
                                                                      'time': float, 'run_id': str, 'run_num': str, 'paradigm': str,
                                                                      'time_var': str, 'HeadX': int, 'HeadY': int,
                                                                      'TailBaseX': int, 'TailBaseY': int, 'Heading': float})
trans_xy_df = trans_xy_df[trans_xy_df['HeadX']+trans_xy_df['HeadY'] > 0]
trans_xy_df = trans_xy_df.reset_index(drop=True)
#dropping index removes data on which 0s got dropped.. ideally this will be fixed with better set up to maximize tracking accuracy
#take results with grain of salt, changes in heading angle over multiple frames will be computed here

cf_runs_df = trans_xy_df[trans_xy_df['paradigm'] == 'cf']
fm_runs_df = trans_xy_df[trans_xy_df['paradigm'] == 'fm']
b1_runs_df = trans_xy_df[trans_xy_df['paradigm'] == 'b1']
b2_runs_df = trans_xy_df[trans_xy_df['paradigm'] == 'b2']

fm_pre_b_df = fm_runs_df[fm_runs_df['time_var'] == 'pre_b']
fm_v_cue_df = fm_runs_df[fm_runs_df['time_var'] == 'v_cue']
fm_gap1_df = fm_runs_df[fm_runs_df['time_var'] == 'gap1']
fm_a_cue_df = fm_runs_df[fm_runs_df['time_var'] == 'a_cue']
fm_gap2_df = fm_runs_df[fm_runs_df['time_var'] == 'gap2']
fm_vid_df = fm_runs_df[fm_runs_df['time_var'] == 'vid']

test_df = fm_a_cue_df[fm_a_cue_df['zf_id'] == 'ZF067']

def calc_ppi(run_data):
    screenside_df = run_data[run_data['HeadX'] > 410]
    farside_df = run_data[run_data['HeadX'] < 210]
    pref_time = sum(screenside_df['frame_time_split'])
    nonpref_time = sum(farside_df['frame_time_split'])
    if pref_time + nonpref_time == 0:
        r_ppi = 'na'
    else:
        print(pref_time, nonpref_time)
        r_ppi = pref_time / (pref_time + nonpref_time)
    return r_ppi

dfs = [cf_runs_df, fm_runs_df]
paradigm_runs_df = pd.concat(dfs)

for fish in list(set(paradigm_runs_df['zf_id'])):
    print('-------', fish, '-------')
    fish_df = paradigm_runs_df[paradigm_runs_df['zf_id'] == fish]
    fish_run_dict = {}
    ppi_dict = {}
    for run in range(1, 19):
        print('--', run, '--')
        run_df = fish_df[fish_df['run_num'] == str(run)]
        run_ppi = calc_ppi(run_df)
        #store run and interval ppis together in a ppi dict
        ppi_dict[fish] = (run_ppi, {'pre_b': [], 'v_cue': [], 'gap1': [], 'a_cue': [], 'gap2': [], 'vid': [], 'post_b': []})
        delta_dict = {'pre_b': [], 'v_cue': [], 'gap1': [], 'a_cue': [], 'gap2': [], 'vid': [], 'post_b': []}
        frame_dict = {'pre_b': [], 'v_cue': [], 'gap1': [], 'a_cue': [], 'gap2': [], 'vid': [], 'post_b': []}
        for i, row in run_df.iterrows():
            if i > run_df.index[0]:
                s = len(run_df) - 1
                h1 = run_df['Heading'][i - 1]
                h2 = run_df['Heading'][i]
                delta = h2 - h1
                paradigm = row['paradigm']
                for k in delta_dict.keys():
                    if k == row['time_var']:
                        frame_dict[k].append(row['frame'])
                        delta_dict[k].append(delta)
        stats_dict = {'pre_b': [], 'v_cue': [], 'gap1': [], 'a_cue': [], 'gap2': [], 'vid': [], 'post_b': []}
        for time_var, deltas in delta_dict.items():
            if len(deltas) > 2:
                avg_delta = sum(deltas) / len(deltas)
                sd = stdev(deltas)
                stats_dict[time_var] = [avg_delta, sd]
                int_df = run_df[run_df['time_var'] == time_var]
                int_ppi = calc_ppi(int_df)
                ppi_dict[fish][1][time_var] = int_ppi
        print(ppi_dict)
'''     po_dict = {}
        for tv in delta_dict.keys():
            delta_counter = 0
            for d, f in zip(delta_dict[tv], frame_dict[tv]):
                if len(stats_dict[tv]) == 0:
                    continue
                if d > stats_dict[tv][0] + stats_dict[tv][1] or d < stats_dict[tv][0] - stats_dict[tv][1]:
                    delta_counter += 1
                    tv_len = len(frame_dict[tv])
                    # print(f, d)
            percent_outlier_deltas = delta_counter / tv_len
            po_dict[tv] = percent_outlier_deltas
        fish_run_dict[run] = po_dict'''
    #plot_pos_dict = {'pre_b': [], 'v_cue': [], 'gap1': [], 'a_cue': [], 'gap2': [], 'vid': [], 'post_b': []}
    #reorganize delta theta data to plot it in a box plot
'''for run,pos_dict in fish_run_dict.items():
        for time_var, pos in pos_dict.items():
            plot_pos_dict[time_var].append(pos)
    pos_df = pd.DataFrame.from_dict(plot_pos_dict, orient='columns')
    b_plot = pos_df.boxplot(column=['pre_b', 'v_cue', 'gap1', 'a_cue', 'gap2', 'vid', 'post_b'])
    if paradigm == 'fm':
        colors = ['black', 'yellow', 'mediumvioletred', 'purple', 'blueviolet', 'mediumblue', 'black']
    if paradigm == 'cf':
        colors = ['black', 'blue', 'mediumvioletred', 'purple', 'blueviolet', 'mediumblue', 'black']

    for i, (d, c)  in enumerate(zip(pos_df, colors)):
        y = pos_df[d]
        x = np.random.normal(i + 1, 0.04, len(y))
        plt.scatter(x, y, color=c)
    plt.title(f'Heading Variance for {fish}, a {paradigm} trial')
    plt.xlabel('Time Interval')
    plt.ylabel('percent of change in heading angles outside of SD')
    # plt.ylim(-10, 50)
    plt.show()
'''
#each run's maximum deviation from average change in direction?





