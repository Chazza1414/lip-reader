import http
import requests
import xml.etree.ElementTree as ET
import os
import glob
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

class webMaus:
    def __init__(self, max_workers=5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.api_counter = 0
        self.lock = threading.Lock()
    
    def increment_api_counter(self):
        with self.lock:
            self.api_counter += 1
            print("Calls: " + self.api_counter)

    def downloadFile(self, download_link, download_type):
        res = requests.get(download_link)
        if (download_type == 'text'):
            return res.text
        elif (download_type == 'csv'):
            return res.text
    
    def runG2P(self, text):
        filedata = {
            'lng': (None, 'eng-GB'),
            'i': ('input.txt', text),
            'iform': (None, 'txt'),
            'oform': (None, 'bpfs'),
            'outsym': (None, 'sampa')
            }
        
        res = requests.post('https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runG2P', files=filedata)

        if (res.status_code == 200):
            root = ET.fromstring(res.text)
            download_link = root.find("downloadLink").text
            success = root.find("success").text

            if(success == 'true' and download_link):
                self.thread_reqs += 1
                print(self.thread_reqs)
                return self.downloadFile(download_link, 'text')
            else:
                raise KeyError("webMAUS error with g2p")
        else:
            raise KeyError("request error with g2p")
    
    def runWebMAUSGeneral(self, audio_file_path, transcription_text):
        filedata = {
            'LANGUAGE': (None, 'eng-GB'),
            'SIGNAL': ('audio.wav', open(audio_file_path, 'rb'), 'wav/WAV'),
            'OUTFORMAT': (None, 'csv'),
            'INSYMBOL': (None, 'sampa'),
            'BPF': ('transcription.par', transcription_text)
            }
        
        res = requests.post('https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUS', files=filedata)

        if (res.status_code == 200):
            root = ET.fromstring(res.text)
            download_link = root.find("downloadLink").text
            success = root.find("success").text

            self.increment_api_counter()

            if(success == 'true' and download_link):
                return self.downloadFile(download_link, 'csv')
                #print(download_link)
            else:
                print(res.text)
                raise KeyError("webMAUS error with g2p")
        else:
            raise KeyError("request error with g2p")
        
    def writePhonemeAlignment(self, csv, write_location):
        sample_rate = 50000
        file_lines = []

        '''
        csv = """BEGIN;DURATION;TOKEN;MAU;MAS;ORT;KAN;TRO;MRP;KAS;SPK;TRN;SPD;VAD
                0;45999;-1;<p:>;;;;;;;;;;
                49000;4499;0;I;;bin;b I n;;;;;;;
                53500;5999;0;n;;bin;b I n;;;;;;;
                59500;1499;1;b;;blue;b l u:;;;;;;;
                61000;2999;1;l;;blue;b l u:;;;;;;;
                64000;2999;1;u:;;blue;b l u:;;;;;;;
                67000;1499;2;@;;at;{ t;;;;;;;
                68500;1499;2;t;;at;{ t;;;;;;;
                70000;5499;3;e;;f;e f;;;;;;;
                81500;6499;4;t;;two;t u:;;;;;;;
                88000;4499;4;u:;;two;t u:;;;;;;;
                92500;3999;5;n;;now;n aU;;;;;;;
                96500;4999;5;aU;;now;n aU;;;;;;;
                101500;48499;-1;<p:>;;;;;;;;;;"""
        '''

        for line in csv.split('\n')[1:]:
            #print(line)
            if line != '':
                line_split = line.split(';')

                if line_split[2] != '-1':
                    start = (int(line_split[0])/sample_rate)*1000
                    end = start+((int(line_split[1])+1)/sample_rate)*1000
                    char = line_split[3]
                    word = line_split[5]

                    file_lines.append(str(start) + ' ' + str(end) + ' ' + char + ' ' + word + '\n')

        with open(write_location, 'w') as file:
            file.writelines(file_lines)

    def runProcess(self, text, audio_file_path, phon_write_location):
        bpfs = self.runG2P(text)
        csv = self.runWebMAUSGeneral(audio_file_path, bpfs)
        self.writePhonemeAlignment(csv, phon_write_location)

    def readFiles(self, audio_file_location, transcription_file_location, phoneme_alignment_location):
        tasks = []
        
        for compressed_path in [d for d in Path(audio_file_location).glob('*') if d.is_dir() and d.suffix != '.tar']:

            #if (os.path.splitext(compressed_path)[0].split('\\')[-1] == 's1_50kHz'):
                #print(compressed_path)
                #print(os.path.splitext(compressed_path)[0].split('\\')[-1])

                for speaker_path in glob.glob(os.path.join(compressed_path, '*')):

                    speaker_id = os.path.splitext(speaker_path)[0].split('\\')[-1]
                    print("speaker: " + speaker_id)
                    n = 1

                    for audio_path in glob.glob(os.path.join(speaker_path, '*')):
                        print(str(n) + "/1000")

                        #print(audio_path)
                        audio_name = os.path.splitext(audio_path)[0].split('\\')[-1]
                        transcription_path = os.path.join(transcription_file_location, speaker_id, 'align', audio_name)
                        transcription_path += '.align'

                        # the following creates a single line containing the transcription of the audio file for the general webMAUS application
                        text = ''
                        with open(transcription_path, 'r') as file:
                            for line in file.readlines()[1:-1]:
                                line_split = line.split(' ')
                                text += line_split[-1] + ' '

                        text = text.strip()
                        text += '\n'

                        #print(text)
                        phon_file_name = speaker_id + '_' + audio_name + '.txt'
                        phon_write_location = os.path.join(phoneme_alignment_location, phon_file_name)

                        tasks.append((text, audio_path, phon_write_location))

                        #self.runProcess(text, audio_path, phon_write_location)

                        n += 1
        with self.executor as executor:  # Ensure proper shutdown
            executor.map(lambda args: self.runProcess(*args), tasks)



wm = webMaus(max_workers=10)
wm.readFiles("H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\audio", 
             "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\transcription", 
             "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\datasets\\phoneme-alignment")
#wm.runProcess("bin blue at f two now\n", "H:/UNI/CS/Year3/Project/Dataset/GRID/audio/s1_50kHz\s1/bbaf2n.wav")