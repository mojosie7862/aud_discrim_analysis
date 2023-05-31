import pandas as pd
from scipy.spatial import distance
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


#calculate ppi over timecouse
def calc_ppi(run_data):
    screenside_df = run_data[run_data['HeadX'] > 330]
    farside_df = run_data[run_data['HeadX'] < 300]
    pref_time = sum(screenside_df['frame_time_split'])
    nonpref_time = sum(farside_df['frame_time_split'])
    if pref_time + nonpref_time == 0:
        ppi = 'na'
    else:
        ppi = pref_time / (pref_time + nonpref_time)
    return ppi


# calculate run place preference indices (ppi) and interval ppis
def get_ppi_df(paradigm_runs, tracking_df):
    run_dfs = {}
    run_nums = ['1', '2', '3', '4', '5', '6']

    run_id_col = []
    run_ppi_col = []
    run_num_col = []
    pre_b_ppi_col = []
    a_cue_ppi_col = []
    gap_ppi_col = []
    vid_ppi_col = []
    post_b_ppi_col = []

    for run in run_nums:
        run_dfs[run] = {}
        for run_id in paradigm_runs:
            if run_id[-1] == run:
                run_dfs[run][run_id] = tracking_df[tracking_df['run_id'] == run_id]

    for run_num, run_dict in run_dfs.items():
        for run_id, data in run_dict.items():
            run_id_col.append(run_id)
            run_num_col.append(run_num)
            run_ppi = calc_ppi(data)
            run_ppi_col.append(run_ppi)
            pre_b_data = data[data['time_var'] == 'pre_b']
            a_cue_data = data[data['time_var'] == 'a_cue']
            gap_data = data[data['time_var'] == 'gap']
            video_data = data[data['time_var'] == 'vid']
            post_b_data = data[data['time_var'] == 'post_b']

            time_var_dfs = [('pre_b', pre_b_data), ('a_cue', a_cue_data), ('gap', gap_data),
                            ('vid', video_data), ('post_b', post_b_data)]

            for tup in time_var_dfs:
                if len(tup[1]) > 0:
                    time_var = tup[1]['time_var'].iloc[0]
                    int_ppi = calc_ppi(tup[1])
                    if time_var == 'pre_b': pre_b_ppi_col.append(int_ppi)
                    if time_var == 'a_cue': a_cue_ppi_col.append(int_ppi)
                    if time_var == 'gap': gap_ppi_col.append(int_ppi)
                    if time_var == 'vid': vid_ppi_col.append(int_ppi)
                    if time_var == 'post_b': post_b_ppi_col.append(int_ppi)
                else:
                    if tup[0] == 'pre_b': pre_b_ppi_col.append('na')
                    if tup[0] == 'a_cue': a_cue_ppi_col.append('na')
                    if tup[0] == 'gap': gap_ppi_col.append('na')
                    if tup[0] == 'vid': vid_ppi_col.append('na')
                    if tup[0] == 'post_b': post_b_ppi_col.append('na')
    ppi_dict = {
        'run_id': run_id_col,
        'run_num': run_num_col,
        'run_ppi': run_ppi_col,
        'pre_b_ppi': pre_b_ppi_col,
        'a_cue_ppi': a_cue_ppi_col,
        'gap_ppi': gap_ppi_col,
        'vid_ppi': vid_ppi_col,
        'post_b_ppi': post_b_ppi_col
    }
    ppi_df = pd.DataFrame.from_dict(ppi_dict)
    return ppi_df


def get_ds_ppi_df(run_nums, paradigm_runs, tracking_df):
    run_dfs = {}


    run_id_col = []
    run_ppi_col = []
    run_num_col = []
    run_paradigm_col = []
    pre_b_ppi_col = []
    a_cue_ppi_col = []
    gap_ppi_col = []
    vid_ppi_col = []
    post_b_ppi_col = []
    a_cue_x_delt_col = []
    gap_x_delt_col = []
    vid_x_delt_col = []
    a_cue_speed_col = []
    gap_speed_col = []
    vid_speed_col = []

    for run in run_nums:
        run_dfs[run] = {}
        for run_id in paradigm_runs:
            run_id_num = run_id[-2:]
            if run_id_num.startswith("r"):
                run_id_num = run_id[-1]
            if run_id_num == run:
                run_dfs[run][run_id] = tracking_df[tracking_df['run_id'] == run_id]

    for run_num, run_dict in run_dfs.items():
        for run_id, data in run_dict.items():
            run_id_col.append(run_id)
            run_num_col.append(run_num)
            run_ppi = calc_ppi(data)
            run_ppi_col.append(run_ppi)
            run_paradigm = data['paradigm'].iloc[0]
            run_paradigm_col.append(run_paradigm)
            pre_b_data = data[data['time_var'] == 'pre_b']
            a_cue_data = data[data['time_var'] == 'a_cue']
            gap_data = data[data['time_var'] == 'gap']
            video_data = data[data['time_var'] == 'vid']
            post_b_data = data[data['time_var'] == 'post_b']

            time_var_dfs = [('pre_b', pre_b_data), ('a_cue', a_cue_data), ('gap', gap_data),
                            ('vid', video_data), ('post_b', post_b_data)]

            for tup in time_var_dfs:
                # if tup[0] == 'a_cue':
                    # a_cue_x_delt = tup[1]['HeadX'].iloc[0]
                    # gap_x_delt = tup[1]['HeadX'].iloc[0]
                    # a_cue_heading_delt = tup[1]['Heading'].iloc[1]-tup[1]['Heading'].iloc[0]
                    # a_cue_x_delt_col.append(a_cue_x_delt)
                    # gap_x_delt_col.append(gap_x_delt)
                if len(tup[1]) > 0:
                    time_var = tup[1]['time_var'].iloc[0]
                    int_ppi = calc_ppi(tup[1])
                    if time_var == 'pre_b': pre_b_ppi_col.append(int_ppi)
                    if time_var == 'a_cue':
                        a_cue_ppi_col.append(int_ppi)
                        a_cue_x_delt = tup[1]['HeadX'].iloc[30]-tup[1]['HeadX'].iloc[0]
                        a_cue_x_delt_col.append(a_cue_x_delt)
                        a_cue_t = sum(tup[1]['frame_time_split'].iloc[:30])
                        a_cue_dist = distance.euclidean(tup[1]['HeadX'].iloc[:30], tup[1]['HeadY'].iloc[:30])
                        a_cue_speed = a_cue_dist/a_cue_t
                        print(run_id)
                        print('a_cue euc dist', a_cue_dist, a_cue_t)
                        print('speed', a_cue_speed)
                        a_cue_speed_col.append(a_cue_speed)
                    if time_var == 'gap':
                        #make function for these parameters ergg
                        gap_ppi_col.append(int_ppi)
                        gap_x_delt = tup[1]['HeadX'].iloc[30]-tup[1]['HeadX'].iloc[0]
                        gap_x_delt_col.append(gap_x_delt)
                        gap_t = sum(tup[1]['frame_time_split'].iloc[:30])
                        gap_dist = distance.euclidean(tup[1]['HeadX'].iloc[:30], tup[1]['HeadY'].iloc[:30])
                        gap_speed = gap_dist / gap_t
                        gap_speed_col.append(gap_speed)
                    if time_var == 'vid':
                        vid_ppi_col.append(int_ppi)
                        vid_x_delt = tup[1]['HeadX'].iloc[30] - tup[1]['HeadX'].iloc[0]
                        vid_x_delt_col.append(vid_x_delt)
                        vid_t = sum(tup[1]['frame_time_split'].iloc[:30])
                        vid_dist = distance.euclidean(tup[1]['HeadX'].iloc[:30], tup[1]['HeadY'].iloc[:30])
                        vid_speed = vid_dist / vid_t
                        vid_speed_col.append(vid_speed)
                    if time_var == 'post_b': post_b_ppi_col.append(int_ppi)
                else:
                    if tup[0] == 'pre_b': pre_b_ppi_col.append('na')
                    if tup[0] == 'a_cue':
                        a_cue_ppi_col.append('na')
                        a_cue_x_delt_col.append('na')
                        a_cue_speed_col.append('na')
                    if tup[0] == 'gap':
                        gap_ppi_col.append('na')
                        gap_x_delt_col.append('na')
                        gap_speed_col.append('na')
                    if tup[0] == 'vid':
                        vid_ppi_col.append('na')
                        vid_x_delt_col.append('na')
                        vid_speed_col.append('na')
                    if tup[0] == 'post_b': post_b_ppi_col.append('na')
    ppi_dict = {
        'run_id': run_id_col,
        'run_num': run_num_col,
        'run_ppi': run_ppi_col,
        'run_paradigm': run_paradigm_col,
        'pre_b_ppi': pre_b_ppi_col,
        'a_cue_ppi': a_cue_ppi_col,
        'gap_ppi': gap_ppi_col,
        'vid_ppi': vid_ppi_col,
        'post_b_ppi': post_b_ppi_col,
        'a_cue_x_delt': a_cue_x_delt_col,
        'gap_x_delt': gap_x_delt_col,
        'vid_x_delt': vid_x_delt_col,
        'a_cue_speed': a_cue_speed_col,
        'gap_speed': gap_speed_col,
        'vid_speed': vid_speed_col
    }
    for v in ppi_dict.values():
        print(len(v))
    ppi_df = pd.DataFrame.from_dict(ppi_dict)
    return ppi_df


sscf_av_tracking_df = import_exp_tracking('tracking_zf_data_10sscf_av.csv')
ssfm_av_tracking_df = import_exp_tracking('tracking_zf_data_10ssfm_av.csv')
ssfm_rew_tracking_df = import_exp_tracking('tracking_zf_data_10ssfm_rew.csv')
sscf_rew_tracking_df = import_exp_tracking('tracking_zf_data_11sscf_rew.csv')
dsfm_av_tracking_df = import_exp_tracking('tracking_zf_data-ZF153-164-12dsfm_av.csv')


cf_runs_df = sscf_rew_tracking_df[sscf_rew_tracking_df['paradigm'] == 'cf']
fm_runs_df = ssfm_av_tracking_df[ssfm_av_tracking_df['paradigm'] == 'fm']
ds_runs_df = pd.concat([cf_runs_df, fm_runs_df])
# b1_runs_df = av_tracking_df[av_tracking_df['paradigm'] == 'b1']
# b2_runs_df = av_tracking_df[av_tracking_df['paradigm'] == 'b2']

cf_ids = set(cf_runs_df['zf_id'])
cf_runs = sorted(list(set(cf_runs_df['run_id'])))
fm_ids = set(fm_runs_df['zf_id'])
fm_runs = sorted(list(set(fm_runs_df['run_id'])))
ds_runs = cf_runs+fm_runs
ds_run_nums = [str(x) for x in range(1,19)]
ss_run_nums = ['1', '2', '3', '4', '5', '6']
ppi_df = get_ds_ppi_df(ss_run_nums, fm_runs, fm_runs_df)
print(ppi_df)
# ppi_df.to_csv('ssfm_av_ppi300-330_delt0-10_velo0-10.csv')




# #dfs with all cf and fm baseline data
# cfb1_baseline_df = b1_runs_df[b1_runs_df['zf_id'].isin(list(cf_ids))]
# cfb2_baseline_df = b2_runs_df[b2_runs_df['zf_id'].isin(list(cf_ids))]
# fmb1_baseline_df = b1_runs_df[b1_runs_df['zf_id'].isin(list(fm_ids))]
# fmb2_baseline_df = b2_runs_df[b2_runs_df['zf_id'].isin(list(fm_ids))]
#
# #lists of baseline run ids by cf/fm and b1/b2
# cf_b1s = list(set(cfb1_baseline_df['run_id']))
# cf_b2s = list(set(cfb2_baseline_df['run_id']))
# fm_b1s = list(set(fmb1_baseline_df['run_id']))
# fm_b2s = list(set(fmb2_baseline_df['run_id']))
#
# #initiating a few variables
# cfrun_dfs = {}
# fmrun_dfs = {}
# cfb1_dfs = {}
# cfb2_dfs = {}
# cf_int_ppi_dict = {}
# cf_run_ppi_dict = {}
# cfb1_ppi = []
# cfb2_ppi = []
# fm_int_ppi_dict = {}
# fm_run_ppi_dict = {}
# fmb1_ppi = []
# fmb2_ppi = []

#split up data frames by paradigm, run, and run_id
# for run in run_nums:
#     cfrun_dfs[run] = {}
#     fmrun_dfs[run] = {}
#     for run_id in cf_runs:
#         if run_id[-1] == run:
#             cfrun_dfs[run][run_id] = av_tracking_df[av_tracking_df['run_id'] == run_id]
#     for run_id in fm_runs:
#         if run_id[-1] == run:
#             fmrun_dfs[run][run_id] = av_tracking_df[av_tracking_df['run_id'] == run_id]

#split by baseline 1 and 2
# for b1 in cf_b1s:
#     cfb1_dfs[b1] = av_tracking_df[av_tracking_df['run_id'] == b1]
# for b2 in cf_b2s:
#     cfb2_dfs[b2] = av_tracking_df[av_tracking_df['run_id'] == b2]


# for run_num, run_dict in cfrun_dfs.items():
#     cf_int_ppi_dict[run_num] = {'run_id':[], 'pre_b': [], 'a_cue': [], 'gap': [], 'vid': [], 'post_b': []}
#     cf_run_ppi_dict[run_num] = []
#     for run_id, data in run_dict.items():
#         cf_run_id_col.append(run_id)
#         cf_run_num_col.append(run_num)
#         cf_run_ppi = calc_ppi(data)
#         cf_run_ppi_dict[run_num].append(cf_run_ppi)
#         cf_run_ppi_col.append(cf_run_ppi)
#         pre_b_data = data[data['time_var'] == 'pre_b']
#         a_cue_data = data[data['time_var'] == 'a_cue']
#         gap_data = data[data['time_var'] == 'gap']
#         video_data = data[data['time_var'] == 'vid']
#         post_b_data = data[data['time_var'] == 'post_b']
#
#         time_var_dfs = [('pre_b', pre_b_data), ('a_cue', a_cue_data), ('gap', gap_data),
#                         ('vid', video_data), ('post_b', post_b_data)]
#
#         for tup in time_var_dfs:
#             if len(tup[1]) > 0:
#                 time_var = tup[1]['time_var'].iloc[0]
#                 zf_id = tup[1]['zf_id'].iloc[0]
#                 cf_int_ppi = calc_ppi(tup[1])
#                 cf_zf_id_col.append(zf_id)
#                 if time_var == 'pre_b':
#                     cf_int_ppi_dict[run_num]['pre_b'].append(cf_int_ppi)
#                     cf_pre_b_ppi_col.append(cf_int_ppi)
#                 if time_var == 'a_cue':
#                     cf_int_ppi_dict[run_num]['a_cue'].append(cf_int_ppi)
#                     cf_a_cue_ppi_col.append(cf_int_ppi)
#                 if time_var == 'gap':
#                     cf_int_ppi_dict[run_num]['gap'].append(cf_int_ppi)
#                     cf_gap_ppi_col.append(cf_int_ppi)
#                 if time_var == 'vid':
#                     cf_int_ppi_dict[run_num]['vid'].append(cf_int_ppi)
#                     cf_vid_ppi_col.append(cf_int_ppi)
#                 if time_var == 'post_b':
#                     cf_int_ppi_dict[run_num]['post_b'].append(cf_int_ppi)
#                     cf_post_b_ppi_col.append(cf_int_ppi)
#             else:
#                 if tup[0] == 'pre_b':
#                     cf_int_ppi_dict[run_num]['pre_b'].append('na')
#                     cf_pre_b_ppi_col.append(cf_int_ppi)
#                 if tup[0] == 'a_cue':
#                     cf_int_ppi_dict[run_num]['a_cue'].append('na')
#                     cf_a_cue_ppi_col.append(cf_int_ppi)
#                 if tup[0] == 'gap':
#                     cf_int_ppi_dict[run_num]['gap'].append('na')
#                     cf_gap_ppi_col.append(cf_int_ppi)
#                 if tup[0] == 'vid':
#                     cf_int_ppi_dict[run_num]['vid'].append('na')
#                     cf_vid_ppi_col.append(cf_int_ppi)
#                 if tup[0] == 'post_b':
#                     cf_int_ppi_dict[run_num]['post_b'].append('na')
#                     cf_post_b_ppi_col.append(cf_int_ppi)
# ppi_dict = {
#     'run_id': cf_run_id_col,
#     'run_num': cf_run_num_col,
#     'run_ppi': cf_run_ppi_col,
#     'pre_b_ppi': cf_pre_b_ppi_col,
#     'a_cue_ppi': cf_a_cue_ppi_col,
#     'gap_ppi': cf_gap_ppi_col,
#     'vid_ppi': cf_vid_ppi_col,
#     'post_b_ppi': cf_post_b_ppi_col
# }
# for v in ppi_dict.values():
#     print(len(v))
# ppi_df = pd.DataFrame.from_dict(ppi_dict)
# ppi_df.to_csv('cf_av_ppi_250-380.csv')
# #calculate FM run place preference indices (ppi) and interval ppis
# for run_num, run_dict in fmrun_dfs.items():
#     fm_int_ppi_dict[run_num] = {'zf_id': [], 'pre_b': [], 'a_cue': [], 'gap': [], 'vid': [], 'post_b': []}
#     fm_run_ppi_dict[run_num] = []
#     for run_id, data in run_dict.items():
#         fm_run_ppi = calc_ppi(data)
#         fm_run_ppi_dict[run_num].append(fm_run_ppi)
#         pre_b_data = data[data['time_var'] == 'pre_b']
#         a_cue_data = data[data['time_var'] == 'a_cue']
#         gap_data = data[data['time_var'] == 'gap']
#         video_data = data[data['time_var'] == 'vid']
#         post_b_data = data[data['time_var'] == 'post_b']
#
#         time_var_dfs = [('pre_b', pre_b_data), ('a_cue', a_cue_data), ('gap', gap_data),
#                         ('vid', video_data), ('post_b', post_b_data)]
#
#         for tup in time_var_dfs:
#             if len(tup[1]) > 0:
#                 time_var = tup[1]['time_var'].iloc[0]
#                 zf_id = tup[1]['zf_id'].iloc[0]
#                 fm_int_ppi = calc_ppi(tup[1])
#                 fm_int_ppi_dict[run_num]['zf_id'].append((zf_id))
#                 if time_var == 'pre_b': fm_int_ppi_dict[run_num]['pre_b'].append(fm_int_ppi)
#                 if time_var == 'a_cue': fm_int_ppi_dict[run_num]['a_cue'].append(fm_int_ppi)
#                 if time_var == 'gap': fm_int_ppi_dict[run_num]['gap'].append(fm_int_ppi)
#                 if time_var == 'vid': fm_int_ppi_dict[run_num]['vid'].append(fm_int_ppi)
#                 if time_var == 'post_b': fm_int_ppi_dict[run_num]['post_b'].append(fm_int_ppi)
#             else:
#                 fm_int_ppi_dict[run_num]['zf_id'].append((zf_id, time_var))
#                 if tup[0] == 'pre_b': fm_int_ppi_dict[run_num]['pre_b'].append('na')
#                 if tup[0] == 'a_cue': fm_int_ppi_dict[run_num]['a_cue'].append('na')
#                 if tup[0] == 'gap': fm_int_ppi_dict[run_num]['gap'].append('na')
#                 if tup[0] == 'vid': fm_int_ppi_dict[run_num]['vid'].append('na')
#                 if tup[0] == 'post_b': fm_int_ppi_dict[run_num]['post_b'].append('na')
#
# print('cf_ids', len(cf_ids), cf_ids)
# print('fm_ids', len(fm_ids), fm_ids)
#
# print('fm_int_ppi')
# for k, v in fm_int_ppi_dict.items():
#     print(k)
#     print(v)
#     # fm_av_pre_avg = sum(v['pre_b'])/len(v['pre_b'])
#     # fm_av_post_avg = sum(v['post_b'])/len(v['post_b'])
#     # print('fm_av_pre_avg', fm_av_pre_avg)
#     # print('fm_av_post_avg', fm_av_post_avg)
#
# print('cf_int_ppi')
# for k, v in cf_int_ppi_dict.items():
#     print(k)
#     print(v)
# print('fm_run_ppi')
# for k, v in fm_run_ppi_dict.items():
#     print(k)
#     print(v)
# print('cf_run_ppi')
# for k, v in cf_run_ppi_dict.items():
#     print(k)
#     print(v)
#
# cf_av_pre_avg = sum(cf_int_ppi_dict['pre_b'])/len(cf_int_ppi_dict['pre_b'])
# cf_av_post_avg = sum(cf_int_ppi_dict['post_b'])/len(cf_int_ppi_dict['post_b'])
#
# print('fm_av_pre_avg', fm_av_pre_avg)
# print('fm_av_post_avg', fm_av_post_avg)
# print('cf_av_pre_avg', cf_av_pre_avg)
# print('cf_av_post_avg', cf_av_post_avg)


# #calculate baseline x position, by strength of bias toward the screen side
# #then calculate relative PPI
#
# print(fm_run_ppi_dict)
# '''#make CF interval ppi dataframe with run_num, global_ppi, and intervals as columns
# cf_run_ppi_df = pd.DataFrame.from_dict(cf_run_ppi_dict, orient='columns')
# b_plot = cf_run_ppi_df.boxplot(column = ['1', '2', '3', '4', '5', '6', '7', '8'])
# b_plot.plot()
#
# for i, d in enumerate(cf_run_ppi_df):
#    y = cf_run_ppi_df[d]
#    x = np.random.normal(i + 1, 0.04, len(y))
#    plt.scatter(x, y)
# plt.title('CF Place Preference Index (PPI)')
# plt.xlabel('Run Number')
# plt.ylabel('PPI')
# #plt.ylim(-10, 50)
# plt.show()'''
#
#
# '''#make FM interval ppi dataframe with run_num, global_ppi, and intervals as columns
# fm_run_ppi_df = pd.DataFrame.from_dict(fm_run_ppi_dict, orient='columns')
# print(fm_run_ppi_df)
# b_plot = fm_run_ppi_df.boxplot(column = ['1', '2', '3', '4', '5', '6', '7', '8'])
# b_plot.plot()
# for i, d in enumerate(fm_run_ppi_df):
#    y = fm_run_ppi_df[d]
#    x = np.random.normal(i + 1, 0.04, len(y))
#    plt.scatter(x, y)
# plt.title('FM Place Preference Index (PPI)')
# plt.xlabel('Run Number')
# plt.ylabel('PPI')
# plt.show()
#
#
#
# for run, int_dict in cf_int_ppi_dict.items():
#     for int, ppi_ls in int_dict.items():
#         for i in ppi_ls:
#             if i == 'na':
#                 ppi_ls.remove('na')
#             else:
#                 type(i) == int
#         int_dict[int] = sum(ppi_ls)/len(ppi_ls)
# int_fig_dict = {'run': [], 'pre_b': [], 'a_cue': [], 'gap': [], 'vid': [], 'post_b': []}
# for run, int_dict in cf_int_ppi_dict.items():
#     int_fig_dict['run'].append(run)
#     for int, ppi_avg in int_dict.items():
#         int_fig_dict[int].append(ppi_avg)
#
# int_fig_df = pd.DataFrame.from_dict(int_fig_dict)
# ints = ['pre_b','a_cue','gap','vid','post_b']
# xs = [1,2,3,4,5,6]
# int_fig_df = int_fig_df[ints]
#
# plt.plot(xs, list(int_fig_df.iloc[0]), label="run 1", linestyle="-", color = 'red', marker='o')
# plt.plot(xs, list(int_fig_df.iloc[1]), label="run 2", linestyle="-", color = 'crimson', marker='o')
# plt.plot(xs, list(int_fig_df.iloc[2]), label="run 3", linestyle="-", color = 'mediumvioletred', marker='o')
# plt.plot(xs, list(int_fig_df.iloc[3]), label="run 4", linestyle="-", color = 'purple', marker='o')
# plt.plot(xs, list(int_fig_df.iloc[4]), label="run 5", linestyle="-", color = 'blueviolet', marker='o')
# plt.plot(xs, list(int_fig_df.iloc[5]), label="run 6", linestyle="-", color = 'indigo', marker='o')
# plt.plot(xs, list(int_fig_df.iloc[6]), label="run 7", linestyle="-", color = 'mediumblue', marker='o')
# plt.plot(xs, list(int_fig_df.iloc[7]), label="run 8", linestyle="-", color = 'black', marker='o')
# plt.xticks(xs, ints)
# plt.legend()
# plt.title('Average PPI Over CF Visual/Auditory Condtitioning Intervals ~ n=6')
# plt.xlabel('Time Interval')
# plt.ylabel('PPI')
# plt.show()
#
#
# int_fig_df.set_index('run')
# b_plot = int_fig_df.boxplot(column=int_fig_df.index)
# b_plot.plot()
# for i, d in enumerate(int_fig_df):
#    y = int_fig_df[d]
#    x = np.random.normal(i + 1, 0.04, len(y))
# groups = int_fig_df.groupby(int_fig_df.index)
# for name, group in groups:
#     plt.plot(group.x, group.y, marker='o', linestyle='', markersize=12, label=int_fig_df.index)
#
# plt.scatter(x, y)
# plt.legend()
# plt.title('CF Place Preference Index (PPI) ~ n=6')
# plt.xlabel('Run Number')
# plt.ylabel('PPI')
# plt.show()
#
#
#
# for b1_id, data in cfb1_dfs.items():
#     b1_run_ppi = calc_ppi(data)
#     cfb1_ppi.append(b1_run_ppi)
# for b2_id, data in cfb2_dfs.items():
#     b2_run_ppi = calc_ppi(data)
#     cfb2_ppi.append(b2_run_ppi)
#
# run_boxpdata = [cfb1_ppi, cfrun_ppi, cfb2_ppi]
# interval_boxpdata = [cfpreb_ppi, cfa_cue_ppi, cfgap_ppi, cfvid_ppi, cfpostb_ppi]
# all_boxpdata = [cfb1_ppi, cfrun_ppi, cfb2_ppi, cfpreb_ppi, cfa_cue_ppi, cfgap_ppi, cfvid_ppi, cfpostb_ppi]
#
#
# names =['b1', 'cf_run', 'b2', 'pre_b', 'a_cue', 'gap', 'vid', 'post_b']
# palette = ['r', 'g', 'b', 'y']
#
# plt.boxplot(all_boxpdata, positions=range(len(all_boxpdata)), labels=names)
# for x, val in zip(all_boxpdata, names):
#     plt.scatter(x, val, alpha=0.4)
# plt.show()
# '''
# #mean and SD of x position for each interval? compare to ppi?
