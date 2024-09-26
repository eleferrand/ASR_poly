# Create partitions
In make_partition.py change the variables lang, text_path and wav_path with the name of the language you are working with and the path to your wav files and the path to your text files

# train Whisper
simply call python train_whisper --data_path data/<lang_name>/split/<max, rand or min>/ --lang <lang_name> --oov_rate <max rand or min>
