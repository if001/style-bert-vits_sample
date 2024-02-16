import numpy as np
import IPython.display

import argparse
import datetime
import json
import os
import sys
from typing import Optional
import torch
import yaml

# 音声ファイル出力
from scipy.io import wavfile

from common.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    GRADIO_THEME,
    LATEST_VERSION,
    Languages,
)
from common.log import logger
from common.tts_model import ModelHolder
from infer import InvalidToneError
from text.japanese import g2kata_tone, kata_tone2phone_tone, text_normalize

class Text2Speech():
    def __init__(self, model_name="jvnv-F2-jp"):
        self.model_name=model_name
        self.kata_tone_json_str = None
        model_dir = "/content/Style-Bert-VITS2/model_assets"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_holder = ModelHolder(model_dir, device)
        self.save_dir = "./tmp/output.wav"

        model_names = self.model_holder.model_names
        if len(model_names) == 0:
            logger.error(
                f"モデルが見つかりませんでした。{model_dir}にモデルを置いてください。"
            )
            sys.exit(1)
        initial_id = 1 ## 0:jvnv-F1-jp, 1: jvnv-F2-jp, 2: jvnv-M2-jp, 3: jvnv-M1-jp
        print(self.model_holder.model_files_dict)
        initial_pth_files = self.model_holder.model_files_dict[model_names[initial_id]]

        # load model
        print(initial_pth_files)
        model_path = initial_pth_files[0]
        print(self.model_name, model_path)
        style, tts_button, speaker = self.model_holder.load_model_gr(self.model_name, model_path)
        # style = DEFAULT_STYLE
        style_weight = DEFAULT_STYLE_WEIGHT
        
        self.tts_fn("準備が出来ました")


    def do(self,to_speak, emotion="Neutral"):
        ## emotion
        #"Neutral": 0,
        #"Angry": 1,
        #"Happy": 2,
        #"Sad": 3

        # style, tts_button, speaker = model_holder.load_model_gr(model_name, model_path)        
        kata_tone = None
        line_split=False

        wrong_tone_message = ""
        kata_tone: Optional[list[tuple[str, int]]] = None
        if self.kata_tone_json_str:
            if line_split:
                logger.warning("Tone generation is not supported for line split.")
                wrong_tone_message = (
                    "アクセント指定は改行で分けて生成を使わない場合のみ対応しています。"
                )
            try:
                kata_tone = []
                json_data = json.loads(self.kata_tone_json_str)
                # tupleを使うように変換
                for kana, tone in json_data:
                    assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
                    kata_tone.append((kana, tone))
            except Exception as e:
                logger.warning(f"Error occurred when parsing kana_tone_json: {e}")
                wrong_tone_message = f"アクセント指定が不正です: {e}"
                kata_tone = None

        # toneは実際に音声合成に代入される際のみnot Noneになる
        tone: Optional[list[int]] = None
        if kata_tone is not None:
            phone_tone = kata_tone2phone_tone(kata_tone)
            tone = [t for _, t in phone_tone]

        speaker_id = self.model_holder.current_model.spk2id[self.model_name]
        print("speaker_id: ", speaker_id)

        start_time = datetime.datetime.now()

        try:
            sr, audio = self.model_holder.current_model.infer(
                text=to_speak,
                language="JP",
                reference_audio_path=None,
                sdp_ratio=DEFAULT_SDP_RATIO,
                noise=DEFAULT_NOISE,
                noisew=DEFAULT_NOISEW,
                length=DEFAULT_LENGTH,
                line_split=DEFAULT_LINE_SPLIT,
                split_interval=DEFAULT_SPLIT_INTERVAL,
                assist_text=None,
                assist_text_weight=DEFAULT_ASSIST_TEXT_WEIGHT,
                use_assist_text=False,
                style= emotion,
                style_weight=DEFAULT_STYLE_WEIGHT,
                given_tone=False,
                sid=speaker_id,
            )            
            wavfile.write(self.save_dir, sr, audio)
        except InvalidToneError as e:
            logger.error(f"Tone error: {e}")
            return f"Error: アクセント指定が不正です:\n{e}", None, kata_tone_json_str
        except ValueError as e:
            logger.error(f"Value error: {e}")
            return f"Error: {e}", None, kata_tone_json_str

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        print('end time', duration)
        return

def main():
    t2s = Text2Speech()
    t2s.tts_fn()

if __name__ == '__main__':
    main()