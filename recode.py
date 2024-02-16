import pyaudio
import wave
def find_device():
    iAudio = pyaudio.PyAudio()
    for x in range(0, iAudio.get_device_count()): 
        print(iAudio.get_device_info_by_index(x))

def recode(save_file, record_sec=4, sampling_rate=44100):
    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format = FORMAT,
        channels = CHANNELS,
        rate = sampling_rate,
        input = True,
        frames_per_buffer = chunk
    )
    
    all = []
    for i in range(0, int(sampling_rate / chunk * record_sec)):
        data = stream.read(chunk)
        all.append(data)
    
    stream.close()   
    p.terminate()
    
    data = b''.join(all)
    
    #保存するファイル名、wは書き込みモード
    out = wave.open(save_file,'w')
    out.setnchannels(1)
    out.setsampwidth(2)
    out.setframerate(sampling_rate)
    out.writeframes(data)
    out.close()

find_device()
# recode('sample.wav')