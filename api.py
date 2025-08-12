# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import time
import os
import re
from typing_extensions import Annotated
from typing import List
from enum import Enum
from io import BytesIO
import logging

import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchaudio
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from fastapi import FastAPI, File, Form, UploadFile

from model import SenseVoiceSmall

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)

TARGET_FS = 16000
model_dir = os.getenv("SENSEVOICE_MODEL_DIR", "iic/SenseVoiceSmall")
device = os.getenv("SENSEVOICE_DEVICE", "cuda:0")

logging.warning(f"---- 使用的 device：{device}; model_dir: {model_dir} ----")

m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=device)
m.eval()

regex = r"<\|.*\|>"

app = FastAPI()


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


@app.post("/api/v1/asr")
async def turn_audio_to_text(
    files: Annotated[List[UploadFile], File(description="wav or mp3 audios in 16KHz")],
    keys: Annotated[str, Form(description="name of each audio joined with comma")] = None,
    lang: Annotated[Language, Form(description="language of audio content")] = "auto",
):
    start_time = time.time()
    audios = []
    for file in files:
        file_io = BytesIO(await file.read())
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)

        # transform to target sample
        if audio_fs != TARGET_FS:
            start_time = time.time()
            resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=TARGET_FS)
            data_or_path_or_list = resampler(data_or_path_or_list)
            logging.info(f"{file.filename} 采样率转换耗时：{time.time() - start_time}")

        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)

    if lang == "":
        lang = "auto"

    if not keys:
        key = [f.filename for f in files]
    else:
        key = keys.split(",")

    res = m.inference(
        data_in=audios,
        language=lang,  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        key=key,
        fs=TARGET_FS,
        **kwargs,
    )
    logging.info(f"推理总耗时：{time.time() - start_time}")

    if len(res) == 0:
        return {"result": []}
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
        it["text"] = rich_transcription_postprocess(it["text"])
    return {"result": res[0]}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SENSEVOICE_HOST", "0.0.0.0")
    port = int(os.getenv("SENSEVOICE_PORT", 9002))

    uvicorn.run(app, host=host, port=port)
