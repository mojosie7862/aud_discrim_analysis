import pandas as pd
from statistics import stdev
import matplotlib.pyplot as plt


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

paradigm_dfs = [cf_runs_df, fm_runs_df]
paradigm_df = pd.concat(paradigm_dfs)
baseline_dfs = [b1_runs_df, b2_runs_df]
baseline_df = pd.concat(baseline_dfs)


#calculate place preference toward the swimming at the wall of the tank closest to the LCD tank
def baseline_ppi(run_data):
    screenside_df = run_data[run_data['HeadX'] > 550]
    farside_df = run_data[run_data['HeadX'] < 550]
    pref_time = sum(screenside_df['frame_time_split'])
    nonpref_time = sum(farside_df['frame_time_split'])
    if pref_time + nonpref_time == 0:
        bl_ppi = 'na'
    else:
        bl_ppi = pref_time / (pref_time + nonpref_time)
    return bl_ppi

#calculate baseline ppi for b1 and b2 of each fish
#calculate baseline ppi for each run for each fish
#look at how it changes between baselines and through the runs

fm = 'fm'
fm_b1_ppis = []
fm_b2_ppis = []
fm_b_delts = []
cf = 'cf'
cf_b1_ppis = []
cf_b2_ppis = []
cf_b_delts = []
zf_ids = list(set(trans_xy_df['zf_id']))

all_tone_x = []
all_tone_y = []
all_gap_x = []
all_gap_y = []
all_video_x = []
all_video_y = []
lengths = []

zfs = []
paradigms = []

for fish in zf_ids:
    print('-------', fish, '-------')
    fish_df = trans_xy_df[trans_xy_df['zf_id'] == fish]
    ppi_dict = {}
    b1_df = fish_df[fish_df['paradigm'] == 'b1']
    b2_df = fish_df[fish_df['paradigm'] == 'b2']
    b1_ppi = baseline_ppi(b1_df)
    b2_ppi = baseline_ppi(b2_df)
    b_delt = b2_ppi - b1_ppi
    if fm in list(set(fish_df['paradigm'])):
        paradigm = fm
        fm_b1_ppis.append(b1_ppi)
        fm_b2_ppis.append(b2_ppi)
        fm_b_delts.append(b_delt)
    if cf in list(set(fish_df['paradigm'])):
        paradigm = cf
        cf_b1_ppis.append(b1_ppi)
        cf_b2_ppis.append(b2_ppi)
        cf_b_delts.append(b_delt)
    # if b_delt < 0:
    #     print(b_delt)
    #     zfs.append(fish)
    #     paradigms.append(paradigm)

    for run in range(1, 7):
        run_df = fish_df[fish_df['run_num'] == str(run)]
        tone_df = run_df[run_df['time_var'] == 'a_cue']
        gap_df = run_df[run_df['time_var'] == 'gap']
        video_df = run_df[run_df['time_var'] == 'vid']
        print(paradigm, 'run', run)

        tone_x = list(tone_df['HeadX'].values)
        tone_y = list(tone_df['HeadY'].values)
        gap_x = list(gap_df['HeadX'].values)
        gap_y = list(gap_df['HeadY'].values)
        video_x = list(video_df['HeadX'].values)
        video_y = list(video_df['HeadY'].values)
        # screen side is
        print('tone', tone_x[19] - tone_x[0], len(tone_x))
        print('gap', gap_x[19] - gap_x[0], len(gap_x))
        print()
        if paradigm == 'fm':
            test = [1,2,3]
            if run in test:
                all_tone_x.append(tone_x)
                all_tone_y.append(tone_y)
                all_gap_x.append(gap_x)
                all_gap_y.append(gap_y)
                all_video_x.append(video_x)
                all_video_y.append(video_y)
        lengths.append((fish, len(gap_x)))

x = [item for sublist in all_tone_x for item in sublist]
y = [item for sublist in all_tone_y for item in sublist]
print('x', len(x))
print('y', len(y))
# print(len(zfs))
print('cf', paradigms.count('cf'))
print('fm', paradigms.count('fm'))
print(lengths)

xlim = 0, 620
ylim = 1, 310

fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))

hb = ax0.hexbin(x, y, gridsize=50, cmap='inferno')
ax0.set(xlim=xlim, ylim=ylim)
ax0.set_title("distribution of fish positions during tone (paradigm=fm, runs 1-3)")
cb = fig.colorbar(hb, ax=ax0, label='counts')

hb = ax1.hexbin(x, y, gridsize=50, bins='log', cmap='inferno')
ax1.set(xlim=xlim, ylim=ylim)
ax1.set_title("With a log color scale")
cb = fig.colorbar(hb, ax=ax1, label='log10(N)')

plt.show()

print('fm_b1s', len(fm_b1_ppis), stdev(fm_b1_ppis))
print('fm_b1 average', sum(fm_b1_ppis)/len(fm_b1_ppis))
print('fm_b2s', len(fm_b2_ppis), stdev(fm_b2_ppis))
print('fm_b2 average', sum(fm_b2_ppis)/len(fm_b2_ppis))
print('fm_b_delts', fm_b_delts)
print()
print('cf_b1s', len(cf_b1_ppis), stdev(cf_b1_ppis))
print('cf_b1 average', sum(cf_b1_ppis)/len(cf_b1_ppis))
print('cf_b2s', len(cf_b2_ppis), stdev(cf_b2_ppis))
print('cf_b2 average', sum(cf_b2_ppis)/len(cf_b2_ppis))
print('cf_b_delts', cf_b_delts)


''' print('--', run, '--')
        b_df = fish_df[fish_df['paradigm'] == str(run)]
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
        print(ppi_dict)'''