import os
import re
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_metric
import torch

oov_rate = "rand"
lang = "Kunwok"


data_path = "/mmfs1/data/leferran/scripts/Polysynthetic/data/{}/split/{}/test/".format(lang, oov_rate)
checkpoint = sorted([x for x in os.listdir("./xlsr53_sub_base_{}/".format(lang+oov_rate)) if "checkpoint" in x], reverse=True)[0]
path_models = "./xlsr53_sub_base_{}/".format(lang+oov_rate)
path_checkpoint = path_models+checkpoint

model = Wav2Vec2ForCTC.from_pretrained(path_checkpoint).to("cuda")
processor = Wav2Vec2Processor.from_pretrained(path_checkpoint)

wer_metric = load_metric("wer")
cer_metric = load_metric("cer")
chars_to_remove_regex = '[\(\)\_\,\?\.\!\-\;\:\"\“\%\‘\”\�\]\[]'

def clean_sent(sent):
    sent = re.sub(chars_to_remove_regex, '', sent).lower()
    return sent

def get_data_reg(data_path):
    data = []
    long_wav = []
    long_transc = ""

    for cpt_w, elt in enumerate(os.listdir(data_path)):

        if ".wav" in elt:
            wav_path = data_path+elt
            with open(data_path+elt.replace(".wav", ".txt"), mode="r", encoding="utf-8") as tfile:
                sent = clean_sent(tfile.read())

            if len(sent.split())>1:
                w, sr = sf.read(wav_path)
     
                long_transc = long_transc+ " "+sent
                
                long_wav = np.concatenate([long_wav, w])
                if (len(long_wav)/sr)>=5:#concatenetion of the corpus to have chunck of at least 23s
                    entry = {}
                    long_transc = long_transc.replace("\n", " ")

                    entry["sentence"] = " ".join(long_transc)
                    entry["audio"] = {"sampling_rate" : sr, "array" : long_wav}
                    data.append(entry)
                    long_wav = []
                    long_transc = ""

                elif (len(long_wav)/sr)<5 :
                    entry = {}
                    long_transc = long_transc.replace("\n", " ")
                    entry["sentence"] = " ".join(long_transc)
                    entry["audio"] = {"sampling_rate" : sr, "array" : long_wav}
                    data.append(entry)
                    long_wav = []
                    long_transc = ""

                if cpt_w==len(list(os.listdir(data_path))):
                    entry = {}
                    long_transc = long_transc.replace("\n", " ")
                    entry["sentence"] = " ".join(long_transc)
                    entry["audio"] = {"sampling_rate" : sr, "array" : long_wav}
                    data.append(entry)
    return data

def prepare_dataset(batch):
    audio = batch["audio"]


    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch
test_data = get_data_reg(data_path)
test_data = list(map(prepare_dataset, test_data))

for ind in range(len(test_data)):
    input_dict = processor(test_data[ind]["input_values"], return_tensors="pt", padding=True)
    logits = model(input_dict.input_values.to("cuda")).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    print(processor.decode(pred_ids))
    print(test_data[ind]["sentence"].lower())