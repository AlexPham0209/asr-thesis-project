import json
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import yt_dlp

dataset_path = os.path.join("data", "raw", "educational_videos")
os.makedirs(dataset_path, exist_ok=True)

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
    
def download_audio(url, path):
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
        'download_archive': os.path.join(dataset_path, 'downloaded.txt'), 
        'outtmpl': os.path.join(path, 'audio.%(ext)s') 
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            metadata = ydl.extract_info(url, download=True)
            return metadata
        except Exception as e:
            return None

def save_transcript_as_json(transcript, metadata, path):
    formatter = JSONFormatter()

    # .format_transcript(transcript) turns the transcript into a JSON string.
    json_formatted = formatter.format_transcript(transcript, indent=2)
    id = metadata.get("id")

    # Writing into json file
    with open(os.path.join(path, 'transcript.json'), 'w', encoding='utf-8') as json_file:
        json_file.write(json_formatted)

def download_video(url, path):
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
        
        'outtmpl': os.path.join(path, '%(title)s.%(ext)s') 
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_sample(url):
    video_id = extract_video_id(url)
    transcript = get_youtube_transcript(url)

    if transcript is None or transcript.is_generated:
        return
    
    path = os.path.join(dataset_path, video_id)
    os.makedirs(path, exist_ok=True)
    metadata = download_audio(url, path)

    if metadata is None:
        return
    
    # Saving metadata in their respective folders
    save_transcript_as_json(transcript, metadata, path)

    with open(os.path.join(path, 'metadata.json'), 'w', encoding='utf-8') as json_file:
        json.dump(metadata, json_file, indent=2)


# Downloading videos
with open(os.path.join(dataset_path, "videos.txt"), "r") as file:
    videos = file.readlines()

for video in videos:
    download_sample(video.strip())