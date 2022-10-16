import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


#this script is dependent on independent variables (iti, # runs, etc.) being standardized across trials/runs beforehand

xy_file = 'xy_results_2022-09-26_14.06.csv'
transcripts = 'combined_transcripts_2022-09-26.csv'
xy_raw_data = pd.read_csv(xy_file)
transcripts_data = pd.read_csv(transcripts)

#match up the xy tracking data with the recording transcript files
critical_frames = {}
critical_data = defaultdict(dict)
for i_v in transcripts_data.index:
    video_filename = transcripts_data['vid_file'][i_v]
    v_zf_id = transcripts_data['fish_id'][i_v]
    run = transcripts_data['run_number'][i_v]
    paradigm = transcripts_data['run_paradigm'][i_v]
    if paradigm == 'cf' or paradigm == 'fm':
        critical_frames = {'paradigm': paradigm,
                           'run': int(run),
                           'cue_start_frame': int(transcripts_data['cue_start_frame'][i_v]),
                           'cue_stop_frame': int(transcripts_data['cue_stop_frame'][i_v]),
                           'tone_start_frame': int(transcripts_data['tone_start_frame'][i_v]),
                           'tone_stop_frame': int(transcripts_data['tone_stop_frame'][i_v]),
                           'video_start_frame': int(transcripts_data['video_start_frame'][i_v]),
                           'video_stop_frame': int(transcripts_data['video_stop_frame'][i_v])}
        critical_data[v_zf_id][run] = {'paradigm': critical_frames['paradigm'],
                                       'pre_cue_xy': [],
                                       'cue_xy': [],
                                       'tone_xy': [],
                                       'pre_vid_xy': [],
                                       'video_xy': [],
                                       'post_vid_xy': [],
                                       'pre_cue_heading': [],
                                       'cue_heading': [],
                                       'tone_heading': [],
                                       'pre_vid_heading': [],
                                       'video_heading': [],
                                       'post_vid_heading': []}
        for i_f in xy_raw_data.index: #over 200,000 iterations
            frame_filename = xy_raw_data['file'][i_f]
            if frame_filename[8:-4] == video_filename[0:-4]:
                # add cue time interval, 2 second interval between sound and video, and post reward/aversion 4s interval
                if xy_raw_data['frame'][i_f] < critical_frames['cue_start_frame']:
                    critical_data[v_zf_id][run]['pre_cue_xy'].append((xy_raw_data['HeadX'][i_f], xy_raw_data['HeadY'][i_f], xy_raw_data['frame'][i_f]))
                    critical_data[v_zf_id][run]['pre_cue_heading'].append(xy_raw_data['Heading'][i_f])
                if critical_frames['cue_start_frame'] <= xy_raw_data['frame'][i_f] <= critical_frames['cue_stop_frame']:
                    critical_data[v_zf_id][run]['cue_xy'].append((xy_raw_data['HeadX'][i_f], xy_raw_data['HeadY'][i_f], xy_raw_data['frame'][i_f]))
                    critical_data[v_zf_id][run]['cue_heading'].append(xy_raw_data['Heading'][i_f])
                if critical_frames['tone_start_frame'] <= xy_raw_data['frame'][i_f] <= critical_frames['tone_stop_frame']:
                    critical_data[v_zf_id][run]['tone_xy'].append((xy_raw_data['HeadX'][i_f], xy_raw_data['HeadY'][i_f], xy_raw_data['frame'][i_f]))
                    critical_data[v_zf_id][run]['tone_heading'].append(xy_raw_data['Heading'][i_f])
                if critical_frames['tone_stop_frame'] <= xy_raw_data['frame'][i_f] <= critical_frames['video_start_frame']:
                    critical_data[v_zf_id][run]['pre_vid_xy'].append((xy_raw_data['HeadX'][i_f], xy_raw_data['HeadY'][i_f], xy_raw_data['frame'][i_f]))
                    critical_data[v_zf_id][run]['pre_vid_heading'].append(xy_raw_data['Heading'][i_f])
                if critical_frames['video_start_frame'] <= xy_raw_data['frame'][i_f] <= critical_frames['video_stop_frame']:
                    critical_data[v_zf_id][run]['video_xy'].append((xy_raw_data['HeadX'][i_f], xy_raw_data['HeadY'][i_f], xy_raw_data['frame'][i_f]))
                    critical_data[v_zf_id][run]['video_heading'].append(xy_raw_data['Heading'][i_f])
                if xy_raw_data['frame'][i_f] > critical_frames['video_stop_frame']:
                    critical_data[v_zf_id][run]['post_vid_xy'].append((xy_raw_data['HeadX'][i_f], xy_raw_data['HeadY'][i_f], xy_raw_data['frame'][i_f]))
                    critical_data[v_zf_id][run]['post_vid_heading'].append(xy_raw_data['Heading'][i_f])

def transform_xy(v_data, c_data, plotx, ploty, plotz, v_xy):
    for r, data in v_data.items():
        c_data[id][r][plotx] = []
        c_data[id][r][ploty] = []
        c_data[id][r][plotz] = []
        for i in data[v_xy]:
            x, y, z = i
            c_data[id][r][plotx].append(x)
            c_data[id][r][ploty].append(y)
            c_data[id][r][plotz].append(z)

for id, videos_data in critical_data.items():
    transform_xy(videos_data, critical_data, 'pre_cue_xdata', 'pre_cue_ydata', 'pre_cue_zdata', 'pre_cue_xy')
    transform_xy(videos_data, critical_data, 'cue_xdata', 'cue_ydata', 'cue_zdata', 'cue_xy')
    transform_xy(videos_data, critical_data, 'tone_xdata', 'tone_ydata', 'tone_zdata', 'tone_xy')
    transform_xy(videos_data, critical_data, 'pre_vid_xdata', 'pre_vid_ydata', 'pre_vid_zdata', 'pre_vid_xy')
    transform_xy(videos_data, critical_data, 'video_xdata', 'video_ydata', 'video_zdata', 'video_xy')
    transform_xy(videos_data, critical_data, 'post_vid_xdata', 'post_vid_ydata', 'post_vid_zdata', 'post_vid_xy')


for r, data in critical_data['ZF046'].items():
    print(r)
    id_run = 'ZF046_r' + r + '_' + data['paradigm']

    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection = '3d')
    ax.plot3D(data['pre_cue_xdata'], data['pre_cue_ydata'], data['pre_cue_zdata'], color = 'tab:gray')
    ax.plot3D(data['cue_xdata'], data['cue_ydata'], data['cue_zdata'], color = 'y')
    ax.plot3D(data['tone_xdata'], data['tone_ydata'], data['tone_zdata'], color = 'r')
    ax.plot3D(data['pre_vid_xdata'], data['pre_vid_ydata'], data['pre_vid_zdata'], color = 'tab:gray')
    ax.plot3D(data['video_xdata'], data['video_ydata'], data['video_zdata'], color = 'b')
    ax.plot3D(data['post_vid_xdata'], data['post_vid_ydata'], data['post_vid_zdata'], color = 'tab:gray')


    #combine all timepoints and differentiate by color

    ax.set_xlabel('x')
    #ax.set_xlim(0, 540)
    ax.set_ylabel('y')
    #ax.set_ylim(0, 260)
    ax.set_zlabel('time (s)')
    ax.set_title(f'{id_run}')

    plt.savefig(f'{id_run}_3d_scatter.png', dpi = 300)
