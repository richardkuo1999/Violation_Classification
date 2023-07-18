import ffmpeg
import subprocess
from pathlib import Path



def get_image(video, save_dir, fps):
    name = video.stem
    save_path = save_dir / name
    save_path.mkdir(exist_ok=True,parents=True)

    input = ffmpeg.input(video)  
    # output_params = ffmpeg.output(input, f'{save_path}/{name}_%d.jpg', r=fps)
    output_params = ffmpeg.output(input, f'{save_path}/%06d.jpg', r=fps)
    ffmpeg.run(output_params)

def get_gps(video, save_dir):
    name = video.stem
    save_path = save_dir / name
    save_path.mkdir(exist_ok=True,parents=True)

    command = f'exiftool -c "%.7f" -n -p "$GPSLatitude,$GPSLongitude" -ee {str(video)} >> {str(save_path/name)}.txt'
    subprocess.run(command, shell=True)

if __name__ == '__main__':
    fps = 1
    source_dir = Path('input')
    save_dir = Path('output')
    for video in source_dir.glob('*.mp4'):
      get_gps(video, save_dir)
      get_image(video, save_dir, fps)
      

