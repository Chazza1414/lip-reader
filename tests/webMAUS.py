import http
import requests
import xml.etree.ElementTree as ET

class webMaus:
    # def __init__(self):
        
    def makeHttpCall(self, url):
        requests.get(url)

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

            if(success == 'true' and download_link):
                return self.downloadFile(download_link, 'csv')
                #print(download_link)
            else:
                print(res.text)
                raise KeyError("webMAUS error with g2p")
        else:
            raise KeyError("request error with g2p")

    def runProcess(self, text, audio_file_path):
        bpfs = self.runG2P(text)
        csv = self.runWebMAUSGeneral(audio_file_path, bpfs)
        print(csv)

wm = webMaus()
wm.runProcess("lay blue at h nine soon\n", "H:/UNI/CS/Year3/Project/lip-reader/GRID/s23_50kHz/s23/lbah9s.wav")