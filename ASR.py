from speechbrain.inference.ASR import EncoderDecoderASR
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="./pretrained_ASR", hparams_file="hyperparams_develop.yaml")
asr_model.transcribe_file("./LibriSpeech/dev-clean-2/1272/135031/1272-135031-0003.flac")