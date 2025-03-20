import speechbrain as sb
import torch
from parse_data import parse_to_json
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.inference.ASR import EncoderDecoderASR

parse_to_json("./LibriSpeech/dev-clean-2")

dataset = DynamicItemDataset.from_json("data.json")

dataset = dataset.filtered_sorted(sort_key="length", select_n=100)

dataset.add_dynamic_item(sb.dataio.dataio.read_audio, takes="file_path", provides="signal")

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="./pretrained_ASR", hparams_file="hyperparams.yaml")

@sb.utils.data_pipeline.takes("words")
@sb.utils.data_pipeline.provides(
        "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
def text_pipeline(words):
      yield words
      tokens_list = asr_model.tokenizer.encode_as_ids(words)
      yield tokens_list
      tokens_bos = torch.LongTensor([asr_model.hparams.bos_index] + (tokens_list))
      yield tokens_bos
      tokens_eos = torch.LongTensor(tokens_list + [asr_model.hparams.eos_index]) # we use same eos and bos indexes as in pretrained model
      yield tokens_eos
      tokens = torch.LongTensor(tokens_list)
      yield tokens

dataset.add_dynamic_item(text_pipeline)

dataset.set_output_keys(["id", "signal", "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens"])
print(dataset[0])