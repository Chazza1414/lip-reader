# LipReader: Phoneme-level Lipreading for Speechless Communication in English

A lipreading neural network trained on the GRID dataset to improve communication with speech-impaired people.

## Usage

To run a pre-trained LipReader model:

1. Clone the project
2. Install the dependencies using the instructions [here](###dependencies)
3. Prepare GPU using [CUDA](https://docs.nvidia.com/cuda/) and [cuDNN](https://docs.nvidia.com/cudnn/)  
**Note:** To not use CUDA change the `tensorflow-gpu` requirement to be `tensorflow` in [requirements](requirements.txt)
4. Run `python -m evaluation.predict`  
**Note:** use `--help` to view command line options

### Dependencies

Steps to install the dependencies required to use LipReader:

1. Install Python 3.10.0
2. Install pip version 25.0.1
3. Install [CMake](https://cmake.org/download/)
4. Install `dlib` compatible with the version of CMake installed using `pip install dlib`
5. Install dependencies from [requirements](requirements.txt) file using:
`pip install -r requirements.txt`

### Constants

## Training

The following steps describe the full process of pre-processing the GRID dataset for training:

### Dataset

From the [GRID](https://spandh.dcs.shef.ac.uk/gridcorpus/) dataset for each speaker download:  
- The raw 50kHz audio
- The high quality video data (pt1 and pt2)
- The word alignments

### Audio Pre-processing

Run `python -m training.webMAUS` with the relevant arguments to create the phoneme alignments  
This script expects all the downloaded `.tar` files to be extracted into a directory for audio, video and transcription  
The extracted files will have a deep directory structure which the script makes use of

### Video Pre-processing

1. Identify the faces in every frame of the videos and create a new video of just the grey lips by running `python -m training.extract_lips`, specifying the location of the video files and where to save them
2. Create symlinks from each lip video to a dataset directory by running `python -m training.prepare`, specifying the location of the lip videos and where to create the symlinks

### Training a Model

From this point most processes used for training rely on the paths defined in [constants](lipreader/common/constants.py)  
These should be adapted to reflect the local repository structure used