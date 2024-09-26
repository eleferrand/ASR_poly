import soundfile as sf
from transformers import WhisperForConditionalGeneration, AutoProcessor
import os, re
from datasets import load_metric
import numpy as np
from tqdm import tqdm
oov_rate = "rand"
lang = "Kunwok"


data_path = "/mmfs1/data/leferran/scripts/Polysynthetic/data/{}/split/{}/test/".format(lang, oov_rate)
checkpoint = max([x for x in os.listdir("whisper_{}_{}/".format(lang,oov_rate)) if "checkpoint" in x], key=lambda y: int(y.split('-')[1]))
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")
print(checkpoint)
print("loading model")
# model = WhisperForConditionalGeneration.from_pretrained("./whisper_{}_{}/checkpoint-3636/".format(lang, oov_rate)).to("cuda") #20
# model = WhisperForConditionalGeneration.from_pretrained("./whisper_{}_{}/checkpoint-4014/".format(lang, oov_rate)).to("cuda") #50
model = WhisperForConditionalGeneration.from_pretrained("./whisper_{}_{}/{}/".format(lang, oov_rate, checkpoint)).to("cuda") #60
processor = AutoProcessor.from_pretrained("openai/whisper-medium")#, language="fr", task="transcribe")
chars_to_remove_regex = '[\(\)\_\,\?\.\!\-\;\:\"\“\%\‘\”\�\>\<]'

data = []
long_wav = []
long_transc = ""

GS_idx = 0
print("load data")
for filename in os.listdir(data_path):
    if ".wav" in filename:
        
        with open(data_path+filename.replace(".wav", ".txt"), mode="r", encoding="utf-8") as tfile:
            transc = tfile.read()
            transc = transc.replace("\n", " ")
            transc = re.sub(chars_to_remove_regex, '', transc).lower()
            # clean transcription
        if len(transc.split())>0:
            long_transc = long_transc+ " "+transc
            w, sr = sf.read(os.path.join(data_path+filename))
            long_wav = np.concatenate([long_wav, w])

            if (len(long_wav)/sr)>=23:#concatenetion of the corpus to have chunck of at least 23s
                entry = {}
                
                entry["sentence"] = long_transc
            
                entry["audio"] = {"sampling_rate" : sr, "array" : long_wav}
                data.append(entry)
                long_wav = []
                long_transc = ""

print("generate predictions")

signals = [x["audio"]["array"] for x in data]
sentences = [x["sentence"] for x in data]

preds = []

for i in tqdm(range(0, len(signals), 10)):
    sig = signals[i:i+10]
    inputs = processor(sig, return_tensors="pt").to("cuda")
    input_features = inputs.input_features

    generated_ids = model.generate(inputs=input_features)
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    preds = preds + list(decoded)


#####ready for eval#####
with open("output_{}{}.txt".format(lang, oov_rate), mode="w", encoding="utf-8") as tfile:
    for i in range(len(preds)):
        pred = preds[i].replace("\n", " ")
        ref = sentences[i].replace("\n", " ")
        tfile.write("prediction:  "+ pred+",")
        tfile.write("reference:   "+ ref+"\n")
        print("prediction:", preds[i])
        print("reference:", sentences[i])
print(wer_metric.compute(predictions=preds, references=sentences))
print(cer_metric.compute(predictions=preds, references=sentences))