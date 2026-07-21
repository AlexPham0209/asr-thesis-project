import json
import os
from pydub import AudioSegment

raw_dataset_path = os.path.join("data", "raw", "educational_videos")
interim_dataset_path = os.path.join("data", "interim")
processed_dataset_path = os.path.join("data", "processed")

raw_audios = [folder for folder in os.listdir(raw_dataset_path) if os.path.isdir(os.path.join(raw_dataset_path, folder))]

for raw_audio_path in raw_audios[:1]:
    print(raw_audio_path)
    audio_path = os.path.join(raw_dataset_path, raw_audio_path, "audio.mp3")
    transcript_path = os.path.join(raw_dataset_path, raw_audio_path, "transcript.json")
    metadata_path = os.path.join(raw_dataset_path, raw_audio_path, "metadata.json")
    
    # Loading in data
    with open(transcript_path, 'r') as file:
        transcript = json.load(file)

    with open(metadata_path, 'r') as file:
        metadata = json.load(file)


    segment = transcript[30]
    start = segment["start"]
    duration = segment["duration"]
    

    # So segmenting is not based on the audio frames but on milliseconds 
    audio = AudioSegment.from_file(audio_path, format="mp3")
    start_time = int(start * 1000)
    end_time = start_time + int(duration * 1000)

    start_time = max(start_time, 0)
    end_time = min(end_time, len(audio))

    segmented_audio = audio[start_time:end_time]

    # Saving segmented audio
    segmented_audio.export('test.mp3', format='mp3')


