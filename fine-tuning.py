import speechbrain as sb
from speechbrain.lobes.features import Fbank
import torch
from speechbrain.inference.ASR import EncoderDecoderASR
from pipeline_prepping import dataset

# Define fine-tuning procedure
class EncDecFineTune(sb.Brain):

    def on_stage_start(self, stage, epoch):
    if stage == sb.Stage.TRAIN:
        print("Modules in self.modules:", self.modules.keys())  # Debug
        for name, module in self.modules.items():
            print(f"Checking module: {name}")  # Debug
            for p in module.parameters():
                p.requires_grad = True

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.signal
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Forward pass
        feats = self.modules.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        #feats.requires_grad = True
        x = self.modules.enc(feats)

        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        return p_seq, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""


        p_seq, wav_lens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss = self.hparams.seq_cost(
            p_seq, tokens_eos, tokens_eos_lens)


        return loss

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="./pretrained_ASR", hparams_file="hyperparams.yaml")

modules = {"enc": asr_model.mods.encoder.model,
           "emb": asr_model.hparams.emb,
           "dec": asr_model.hparams.dec,
           "compute_features": asr_model.mods.encoder.compute_features, # we use the same features
           "normalize": asr_model.mods.encoder.normalize,
           "seq_lin": asr_model.hparams.seq_lin,

          }

hparams = {"seq_cost": lambda x, y, z: sb.nnet.losses.nll_loss(x, y, z, label_smoothing = 0.1),
            "log_softmax": sb.nnet.activations.Softmax(apply_log=True)}

brain = EncDecFineTune(modules, hparams=hparams, opt_class=lambda x: torch.optim.SGD(x, 1e-5))
brain.tokenizer = asr_model.tokenizer

brain.fit(range(2), train_set=dataset,
          train_loader_kwargs={"batch_size": 8, "drop_last":True, "shuffle": False})