import os
import glob
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.constants import VIDEO_PATH, DATASET_PATH

'''
Usage: 
$ python prepare.py [Path to video dataset] [Path to align dataset] [Number of samples]

where the number of samples is the number of speakers to be reserved for validation

iterate through every video file tree
create a symlink from the video to the dataset folder
renaming the file to be 's[speaker_number]_[video_name]'
'''

VALIDATION_SAMPLES = int(sys.argv[3])

for compressed_path in glob.glob(os.path.join(VIDEO_PATH, '*'))+glob.glob(os.path.join(VIDEO_PATH, '*.part2')):
    print(os.path.splitext(compressed_path)[0].split('/')[-1])
    for speaker_path in glob.glob(os.path.join(compressed_path, '*')):
        speaker_id = os.path.splitext(speaker_path)[0].split('/')[-1]

        n = 0
        for video_path in glob.glob(os.path.join(speaker_path, 'video', 'mpg_6000', '*')):
            video_name = os.path.splitext(video_path)[0].split('/')[-1]
            # does this include the extension - no

            if n < VALIDATION_SAMPLES:
                subprocess.check_output(
                    "ln -s '{}' '{}'".format(video_path, os.path.join(DATASET_PATH, 'validate', speaker_id + "_" + video_name)), shell=True)
            else:
                subprocess.check_output(
                    "ln -s '{}' '{}'".format(video_path, os.path.join(DATASET_PATH, 'train', speaker_id + "_" + video_name)), shell=True)
            n += 1


'''
DATASET_VIDEO_PATH = sys.argv[1]
DATASET_ALIGN_PATH = sys.argv[2]

VAL_SAMPLES = int(sys.argv[3])

for speaker_path in glob.glob(os.path.join(DATASET_VIDEO_PATH, '*')):
    speaker_id = os.path.splitext(speaker_path)[0].split('/')[-1]

    subprocess.check_output("mkdir -p '{}'".format(os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'train')), shell=True)

    for s_path in glob.glob(os.path.join(DATASET_VIDEO_PATH, '*')):
        s_id = os.path.splitext(s_path)[0].split('/')[-1]

        if s_path == speaker_path:
            subprocess.check_output("mkdir -p '{}'".format(os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'train', s_id)), shell=True)
            subprocess.check_output("mkdir -p '{}'".format(os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'val', s_id)), shell=True)
            n = 0
            for video_path in glob.glob(os.path.join(DATASET_VIDEO_PATH, speaker_id, '*')):
                video_id = os.path.splitext(video_path)[0].split('/')[-1]
                if n < VAL_SAMPLES:
                    subprocess.check_output("ln -s '{}' '{}'".format(video_path, os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'val', s_id, video_id)), shell=True)
                else:
                    subprocess.check_output("ln -s '{}' '{}'".format(video_path, os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'train', s_id, video_id)), shell=True)
                n += 1
        else:
            subprocess.check_output("ln -s '{}' '{}'".format(s_path, os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'train', s_id)), shell=True)
    subprocess.check_output("ln -s '{}' '{}'".format(DATASET_ALIGN_PATH, os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'align')), shell=True)
'''