from pathlib import Path
i = 0
a=0
for video in Path('ok').iterdir():
    for file in video.glob('*.jpg'):
        filename = video / f'{file.stem}.txt'
        if not filename.exists():
            with open(filename, 'a'):
                pass