import os
import glob
import subprocess
import sys
import tarfile
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lipreader.common.constants import VIDEO_PATH, DATASET_PATH, VALIDATION_SAMPLES

'''
Usage: 
$ python prepare.py

where the number of samples is the number of speakers to be reserved for validation

iterate through every video file tree
create a symlink from the video to the dataset folder
renaming the file to be 's[speaker_number]_[video_name]'
'''

n = 0
# for compressed_path in [d for d in 
#                         (Path(os.path.join(VIDEO_PATH, '*'), os.path.join(VIDEO_PATH, '*.part2'))).glob('*') 
#                         if d.is_dir() and d.suffix != '.tar']:
#print([d for d in glob.glob(VIDEO_PATH)])
#for compressed_path in [d for d in glob.glob(VIDEO_PATH + '/*') if not tarfile.is_tarfile(d)]:
for compressed_path in [d for d in Path(VIDEO_PATH).glob('*')]:
#for compressed_path in glob.glob(os.path.join(VIDEO_PATH, '*')):
    print(os.path.splitext(compressed_path)[0].split('\\')[-1])
    for speaker_path in Path(compressed_path).glob('*'):
        speaker_id = os.path.splitext(speaker_path)[0].split('\\')[-1]

        for video_path1 in Path(speaker_path).glob('*'):
            for video_path2 in Path(video_path1).glob('*'):
                for video_path3 in Path(video_path2).glob('*'):

                    video_name = os.path.splitext(video_path3)[0].split('\\')[-1]
                    # does this include the extension - no

                    # print(video_path3, os.path.join(DATASET_PATH, 'validate', speaker_id + "_" + video_name))
                    # print(speaker_id)
                    #print()
                    # link_name = Path(DATASET_PATH) / 'validate' / (speaker_id + "_" + video_name + '.mpg')
                    # print(link_name)

                    validate_link_name = Path(DATASET_PATH) / 'validate' / (speaker_id + "_" + video_name + '.mpg')
                    train_link_name = Path(DATASET_PATH) / 'train' / (speaker_id + "_" + video_name + '.mpg')

                    if os.path.exists(validate_link_name):
                        os.remove(validate_link_name)  # Remove existing link or file
                    if os.path.exists(train_link_name):
                        os.remove(train_link_name)  # Remove existing link or file
                        
                    try:
                        if n < VALIDATION_SAMPLES:
                            os.symlink(video_path3, validate_link_name)
                            # subprocess.check_output(
                            #     "mklink {} {}".format(link_name, video_path3), shell=True)
                        else:
                            os.symlink(video_path3, train_link_name)
                            # subprocess.check_output(
                            #     "mklink {} {}".format(link_name, video_path3), shell=True)
                    
                    except Exception as error:
                        raise(error)
                    


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