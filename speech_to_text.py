from faster_whisper import WhisperModel

class Speech2Txt():
    def __init__(self):
        model_size = "small"
        # model_size = "base"
        # model_size = "large-v2"

        device="auto"
        # device="cuda"
        self.wpmodel = WhisperModel(model_size, device=device, compute_type="float16")

    def do(self, file_path):
        segments, info = self.wpmodel.transcribe(file_path, beam_size=5)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        text = ""
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            text = text + segment.text
        print(text)
        return text

    
