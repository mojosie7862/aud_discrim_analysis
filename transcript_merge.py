import os
import pandas as pd
from datetime import datetime
from collections import defaultdict

#combine trial transcript data from folders inside the 'video_data' folder

root = 'video_data'

dfs = []

for item in os.listdir(root):
    trial_file_path = root + '/' + item
    for i in os.listdir(trial_file_path):
        if i.startswith('transcript'):
            transcript_file_path = os.path.join(trial_file_path, i)
            df = pd.read_csv(transcript_file_path)
            dfs.append(df)
print('# of trial transcript files combined:', len(dfs))
df = pd.concat(dfs)
df['r_id'] = df['run_id'].astype(str)
df = df.drop('run_id', axis='columns')
transcripts_data = df.set_index('r_id')
transcripts_data.index.names = ['run_id']
date = str(datetime.now().strftime('%Y-%m-%d'))
filename = 'combined_transcripts_'+date+'.csv'
transcripts_data.to_csv(filename)
