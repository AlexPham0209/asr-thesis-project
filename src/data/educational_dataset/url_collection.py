import os

from selenium import webdriver
from selenium.webdriver import ChromeOptions
import yt_dlp

def get_videos_from_playlist(playlist_url):
    ydl_opts = {
        'extract_flat': True,  # Only extract metadata, do not download media
        'skip_download': True, # Explicitly skip downloading actual files
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(playlist_url, download=False)

        if 'entries' not in result:
            return
        
        videos = result['entries']

        return videos


# Creating dataset folder if it doesn't exist
dataset_path = os.path.join("data", "raw", "educational_videos")
os.makedirs(dataset_path, exist_ok=True)

with open(os.path.join(dataset_path, "playlists.txt"), "r") as file:
    playlists = file.readlines()


urls = []
for playlist in playlists:
    videos = get_videos_from_playlist(playlist.strip())
    playlist_urls = list(map(lambda video: video["url"], videos))
    urls.extend(playlist_urls)

# Writing all video urls in file
with open(os.path.join(dataset_path, "videos.txt"), "w") as f:
    f.write("\n".join(urls))

