import json
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import yt_dlp

dataset_path = os.path.join("data", "raw", "educational_videos")
videos_path = os.path.join(dataset_path, "videos")
audios_path = os.path.join(dataset_path, "audio")
metadata_path = os.path.join(dataset_path, "metadata")
transcripts_path = os.path.join(dataset_path, "transcripts")

os.makedirs(dataset_path, exist_ok=True)
os.makedirs(videos_path, exist_ok=True)
os.makedirs(audios_path, exist_ok=True)
os.makedirs(metadata_path, exist_ok=True)
os.makedirs(transcripts_path, exist_ok=True)

def extract_video_id(url):
    """Extracts the 11-character video ID from various YouTube URL formats."""
    pattern = r'(?:v=|\/embed\/|\/1\/|\/v\/|https:\/\/youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

def get_youtube_transcript(url):
    try:
        # Step 1: Get the video ID
        video_id = extract_video_id(url)
        
        # Step 2: Fetch the transcript data (list of dictionaries)
        ytt_api = YouTubeTranscriptApi()
        raw_transcript = ytt_api.fetch(video_id)
        
        return raw_transcript

    except Exception as e:
        return None
    
def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],

        'postprocessor_args': [
            '-ar', '16000'               
        ],
        
        'noplaylist': True,
        'download_archive': os.path.join(audios_path, 'downloaded.txt'), 
        'outtmpl': os.path.join(audios_path, '%(id)s.%(ext)s') 
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            metadata = ydl.extract_info(url, download=True)
            return metadata
        except Exception as e:
            return None

def save_transcript_as_json(transcript, metadata):
    formatter = JSONFormatter()

    # .format_transcript(transcript) turns the transcript into a JSON string.
    json_formatted = formatter.format_transcript(transcript, indent=2)
    id = metadata.get("id")

    # Writing into json file
    with open(os.path.join(transcripts_path, f'{id}.json'), 'w', encoding='utf-8') as json_file:
        json_file.write(json_formatted)

def download_video(url):
    ydl_opts = {
        'format': 'bestaudio',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],

        'postprocessor_args': [
            '-ar', '16000'               
        ],
        
        'outtmpl': os.path.join(videos_path, '%(title)s.%(ext)s') 
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_sample(url):
    transcript = get_youtube_transcript(url)

    if transcript is None or transcript.is_generated:
        return
    
    metadata = download_audio(url)
    
    # Saving metadata in their respective folders
    save_transcript_as_json(transcript, metadata)

    with open(os.path.join(metadata_path, f'{metadata.get("id")}.json'), 'w', encoding='utf-8') as json_file:
        json.dump(metadata, json_file, indent=2)


download_sample("https://www.youtube.com/watch?v=Qf8hIBrsWZI")
