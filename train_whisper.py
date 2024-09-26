from transformers import Seq2SeqTrainer
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from argparse import ArgumentParser
from praatio import tgio
import soundfile as sf
import os, re
import numpy as np
import argparse
import csv
from memory_profiler import profile

print("grabbing models")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium", task="transcribe")
metric = evaluate.load("wer")
chars_to_remove_regex = '[\(\)\_\,\?\.\!\-\;\:\"\“\%\‘\”\�\]\[]'

def clean_sent(sent):
    sent = re.sub(chars_to_remove_regex, '', sent).lower()
    return sent


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

def get_data_conc(data_path):
    data = []
    long_wav = []
    long_transc = ""
    duration = 0
    words = []

    for cpt_w, elt in enumerate(os.listdir(data_path)):

        if ".wav" in elt:
            wav_path = data_path+elt
            with open(data_path+elt.replace(".wav", ".txt"), mode="r", encoding="utf-8") as tfile:
                sent = clean_sent(tfile.read())

            if len(sent.split())>1:
                w, sr = sf.read(wav_path)
     
                long_transc = long_transc+ " "+sent
                
                long_wav = np.concatenate([long_wav, w])
                if (len(long_wav)/sr)>=20:#concatenetion of the corpus to have chunck of at least 23s
                    entry = {}
                    duration+=len(long_wav)
                    words = words+long_transc.split()
                    entry["sentence"] = long_transc.replace("\n", " ")
                
                    entry["audio"] = {"sampling_rate" : sr, "array" : long_wav}
                    data.append(entry)
                    # if len(long_wav)/sr>20:
                    #     print("oh oh")
                    long_wav = []
                    long_transc = ""
                # elif (len(long_wav)/sr)<20 :
                #     entry = {}
                #     duration+=len(long_wav)
                #     words = words+long_transc.split()
                #     entry["sentence"] = long_transc.replace("\n", " ")
                
                #     entry["audio"] = {"sampling_rate" : sr, "array" : long_wav}
                #     data.append(entry)
                #     if len(long_wav)/sr>30:
                #         print("oh oh")
                #     long_wav = []
                #     long_transc = ""
                if cpt_w==len(list(os.listdir(data_path))):
                    entry = {}
                    duration+=len(long_wav)
                    words = words+long_transc.split()
                    entry["sentence"] = long_transc.replace("\n", " ")
                    entry["audio"] = {"sampling_rate" : sr, "array" : long_wav}
                    data.append(entry)
            
    print(len(words))
    print(len(set(words)))
    print(duration/16000)
    return data

def get_data(data_path):
    data = []

    duration = 0
    words = []
    
    for elt in os.listdir(data_path):
        if ".wav" in elt:
            entry={}
            wav_path = data_path+elt
            with open(data_path+elt.replace(".wav", ".txt"), mode="r", encoding="utf-8") as tfile:
                sent = clean_sent(tfile.read())
            words = words+sent.split()
            
            w, sr = sf.read(wav_path)
            duration+=len(w)

            entry["sentence"] = sent
        
            entry["audio"] = {"sampling_rate" : sr, "array" : w}
            data.append(entry)

            
    print(len(words))
    print(len(set(words)))
    print(duration/16000)
    return data


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
@profile
def train(data_path, lang, oov_rate):
    print("loading data")
    train_data = list(map(prepare_dataset, get_data_conc(data_path+"train/"))) 
    dev_data = list(map(prepare_dataset, get_data_conc(data_path+"test/")))
    for elt in dev_data:
        print(len(elt["audio"]["array"])/elt["audio"]["sampling_rate"])

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    print("initializing training")
    epochs = 15
    batch_size = 8
    nb_samples = len(train_data)
    steps = int((nb_samples/batch_size)*epochs)
    print(steps)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper_{}_{}".format(lang, oov_rate),  # change to a repo name of your choice
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        max_steps=steps,#4000
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=round(steps/10),
        eval_steps=round(steps/10),
        warmup_steps = round(steps/10),
        save_total_limit=2,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    print("training")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=dev_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor)

    trainer.train()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="smp")
    parser.add_argument("--lang", type=str, default="smp")
    parser.add_argument("--oov", type=str, default="smp")

    #your data path should contain a train and test folders with inside a set of wavs and txts files
    args = parser.parse_args()
    data_path = args.data_path
    lang = args.lang
    oov_rate = args.oov

    train(data_path, lang, oov_rate)

if __name__ == "__main__":
    main()