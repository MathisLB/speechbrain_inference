from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.utils.parameter_transfer import Pretrainer

model = ECAPA_TDNN(input_size= 80,
                   channels= [1024, 1024, 1024, 1024, 3072],
                   kernel_sizes= [5, 3, 3, 3, 1],
                   dilations= [1, 2, 3, 4, 1],
                   attention_channels= 128,
                   lin_neurons = 192)

pretrain = Pretrainer(loadables={'model': model}, paths={'model': 'speechbrain/spkrec-ecapa-voxceleb/embedding_model.ckpt'})
pretrain.collect_files()
pretrain.load_collected()