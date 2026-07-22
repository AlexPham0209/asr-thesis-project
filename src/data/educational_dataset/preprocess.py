import json
import os
from pydub import AudioSegment
import pandas as pd
import re

raw_dataset_path = os.path.join("data", "raw", "educational_videos")
interim_dataset_path = os.path.join("data", "interim", "educational_videos")
processed_dataset_path = os.path.join("data", "processed")

audios_path = os.path.join(interim_dataset_path, "audios")
os.makedirs(audios_path, exist_ok=True)

raw_audios = [folder for folder in os.listdir(raw_dataset_path) if os.path.isdir(os.path.join(raw_dataset_path, folder))]

audios = []
texts = []

for name in raw_audios:
    audio_path = os.path.join(raw_dataset_path, name, "audio.mp3")
    transcript_path = os.path.join(raw_dataset_path, name, "transcript.json")
    metadata_path = os.path.join(raw_dataset_path, name, "metadata.json")

    # Loading in data
    with open(transcript_path, 'r') as file:
        transcript = json.load(file)
    
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    for i, segment in enumerate(transcript):
        text = segment["text"]
        start = segment["start"]
        duration = segment["duration"]

        # So segmenting is not based on the audio frames but on milliseconds 
        audio = AudioSegment.from_file(audio_path, format="mp3")
        start_time = int(start * 1000)
        end_time = start_time + int(duration * 1000)
        
        # Replacing all new lines 
        text = re.sub(r'[\n\t]', ' ', text)

        start_time = max(start_time, 0)
        end_time = min(end_time, len(audio))
        
        # Saving segmented audio
        file_name = f"{name}_{i}.mp3"
        segmented_audio = audio[start_time:end_time]
        segmented_audio.export(os.path.join(audios_path, file_name), format='mp3')

        # Adding to lists
        audios.append(os.path.join("audios", file_name))
        texts.append(text)


# Saving data as a csv
data = {
    "text": texts,
    "audio_path": audios
}

df = pd.DataFrame(data)
df.to_csv(os.path.join(interim_dataset_path, "dataset.csv"), index=False)




