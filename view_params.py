from speechbrain.inference.ASR import EncoderDecoderASR
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="./pretrained_ASR", hparams_file="hyperparams.yaml")

print(asr_model.mods.keys())

print(asr_model.mods.encoder)

print(asr_model.mods.encoder.compute_features)

print(dir(asr_model.hparams))