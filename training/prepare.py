import os, argparse
from pathlib import Path
from lipreader.common.constants import DATASET_PATH, VALIDATION_FRACTION, MAX_NUM_VIDEOS, LIPS_PATH, EVALUATION_FRACTION

'''
Usage: 
$ python prepare.py

where the number of samples is the number of speakers to be reserved for validation

iterate through every video file tree
create a symlink from the video to the dataset folder
renaming the file to be 's[speaker_number]_[video_name]'
'''

def create_video_links(lips_path=LIPS_PATH, dataset_path=DATASET_PATH):
    validation_number = int(VALIDATION_FRACTION * int(MAX_NUM_VIDEOS))
    evaluation_number = int(EVALUATION_FRACTION * int(MAX_NUM_VIDEOS))

    n = 0
    for video_path in Path(lips_path).glob('*'):

        speaker_id = os.path.splitext(video_path)[0].split('\\')[-1].split("_")[0]
        video_name = os.path.splitext(video_path)[0].split('\\')[-1].split("_")[1]

        validate_link_name = Path(dataset_path) / 'validate' / (speaker_id + "_" + video_name + '.mpg')
        train_link_name = Path(dataset_path) / 'train' / (speaker_id + "_" + video_name + '.mpg')
        evaluate_link_name = Path(dataset_path) / 'evaluate' / (speaker_id + "_" + video_name + '.mpg')

        if os.path.exists(validate_link_name):
            os.remove(validate_link_name)
        if os.path.exists(train_link_name):
            os.remove(train_link_name)
        if os.path.exists(evaluate_link_name):
            os.remove(evaluate_link_name)
            
        try:
            print("creating link " + str(n))
            if n < validation_number:
                os.symlink(video_path, validate_link_name)
            elif n < (validation_number + evaluation_number):
                os.symlink(video_path, evaluate_link_name)
            else:
                os.symlink(video_path, train_link_name)
        
        except Exception as error:
            raise(error)
        n += 1

def prepare_test_videos(input_video_location, output_dataset_location, max_num_videos):
    validation_number = int(VALIDATION_FRACTION * int(max_num_videos))
    n = 0

    for video_path in Path(input_video_location).glob('*'):

        speaker_id = os.path.splitext(video_path)[0].split('\\')[-1].split("_")[0]

        video_name = os.path.splitext(video_path)[0].split('\\')[-1].split("_")[1]
        
        validate_link_name = Path(output_dataset_location) / 'validate' / (speaker_id + "_" + video_name + '.mpg')
        train_link_name = Path(output_dataset_location) / 'train' / (speaker_id + "_" + video_name + '.mpg')

        if os.path.exists(validate_link_name):
            os.remove(validate_link_name)  # Remove existing link or file
        if os.path.exists(train_link_name):
            os.remove(train_link_name)  # Remove existing link or file
            
        try:
            print("writing test file")
            if n < validation_number:
                os.symlink(video_path, validate_link_name)
                # subprocess.check_output(
                #     "mklink {} {}".format(link_name, video_path3), shell=True)
            else:
                os.symlink(video_path, train_link_name)
                # subprocess.check_output(
                #     "mklink {} {}".format(link_name, video_path3), shell=True)
        
        except Exception as error:
            raise(error)
        n += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='extract_lips',
        description='Pre-processes video data to extract lips'
    )

    parser.add_argument('lips', help='location of the lip videos')
    parser.add_argument('dataset', help='location of the dataset to save the symlinks to')

    args = parser.parse_args()

    create_video_links(args.lips, args.dataset)

# vid_path = "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\video\\s6.mpg_6000.part2\\s6\\video\\mpg_6000\\pbab5n.mpg"

# prepare videos normally
# create_video_links()

# prepare test subset
#prepare_videos()
#prepare_test_videos("H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\test_video", "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\test_datasets", 11)