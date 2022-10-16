import os
import pandas as pd
from datetime import datetime

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
date = str(datetime.now().strftime('%Y-%m-%d'))
filename = 'combined_transcripts_'+date+'.csv'
df.to_csv(filename)


