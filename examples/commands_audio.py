from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow,seed=1208251813'

import numpy as np

from odin import fuel as F, nnet as N, backend as K, training
from odin.utils import get_all_files, get_datasetpath, get_modelpath
from odin.stats import freqcount
from odin.basic import has_roles, WEIGHT, BIAS

# ===========================================================================
# Const
# ===========================================================================
FEAT = 'mspec' # using mel-spectrogram
np.random.seed(12082518)

# ===========================================================================
# Load wav files
# ===========================================================================
wav_path = F.load_commands_wav()
print("Found:",
      len(get_all_files(wav_path, filter_func=lambda x: '.wav' in x)),
      " .wav files")
datapath = get_datasetpath("commands", override=False)
# ====== start preprocessing audio files ====== #
if False:
    speech = F.SpeechProcessor(wav_path, datapath,
                win=0.025, shift=0.01, nb_melfilters=40, nb_ceps=13,
                get_spec=True, get_mspec=True, get_mfcc=True,
                get_qspec=True, get_phase=True, get_pitch=False,
                get_energy=True, get_delta=2, fmin=64, fmax=4000,
                sr_new=16000, preemphasis=0.97,
                pitch_threshold=0.8, pitch_fmax=800,
                get_vad=2, vad_smooth=3, vad_minlen=0.1,
                cqt_bins=96, pca=True, pca_whiten=False,
                center=True, audio_ext='.wav',
                save_stats=True, substitute_nan=None,
                dtype='float16', datatype='memmap',
                ncache=0.12, ncpu=8)
    speech.run()
# ====== load the dataset ====== #
ds = F.Dataset(datapath, read_only=True)
print(ds)
# ===========================================================================
# Split the dataset for train, valid, test
# ===========================================================================
indices = sorted(
    ((name, start, end) for name, (start, end) in ds['indices'].iteritems()),
    key=lambda x: x[0]
)
all_labels = list(set(i[0].split("_")[0] for i in indices))
print("Labels:", all_labels)
np.random.shuffle(indices); np.random.shuffle(indices)
longest_utterance = max(int(end - start) for name, start, end in indices)
print("Longest Utterance:", longest_utterance)
nb_files = len(indices)

train_indices = indices[:int(0.6 * nb_files)]
valid_indices = indices[int(0.6 * nb_files):int(0.8 * nb_files)]
test_indices = indices[int(0.8 * nb_files):]

print("Train distribution:", len(train_indices),
    freqcount([x[0].split('_')[0] for x in train_indices]).items())
print("Valid distribution:", len(valid_indices),
    freqcount([x[0].split('_')[0] for x in valid_indices]).items())
print("Test distribution:", len(test_indices),
    freqcount([x[0].split('_')[0] for x in test_indices]).items())

train = F.Feeder(ds[FEAT], train_indices, ncpu=1)
valid = F.Feeder(ds[FEAT], valid_indices, ncpu=1)
test = F.Feeder(ds[FEAT], test_indices, ncpu=1)

recipes = [
    F.recipes.Name2Trans(
        converter_func=lambda x: all_labels.index(x.split("_")[0])
    ),
    F.recipes.Normalization(
        mean=ds[FEAT + "_mean"],
        std=ds[FEAT + "_std"],
        local_normalize=False
    ),
    F.recipes.Sequencing(frame_length=longest_utterance, hop_length=1,
                         end='pad', endmode='post'),
    F.recipes.CreateFile(return_name=False)
]

train.set_recipes(recipes)
valid.set_recipes(recipes)
test.set_recipes(recipes)

print("Train shape:", train.shape)
print("Valid shape:", valid.shape)
print("Test shape:", test.shape)

# ===========================================================================
# Create the network
# ===========================================================================
print('Feature shape:', train.shape)
feat_shape = (None,) + train.shape[1:]
X = K.placeholder(shape=feat_shape, name='X')
y = K.placeholder(shape=(None,), dtype='int32', name='y')

f = N.Sequence([
    # # ====== CNN ====== #
    N.Dimshuffle(pattern=(0, 1, 2, 'x')),
    N.Conv(num_filters=32, filter_size=3, pad='same', strides=1,
           activation=K.linear),
    N.BatchNorm(activation=K.relu),
    N.Flatten(outdim=3),

    # ====== RNN ====== #
    N.AutoRNN(64, rnn_mode='lstm', num_layers=1,
              direction_mode='unidirectional', prefer_cudnn=True),

    # ====== Dense ====== #
    N.Flatten(outdim=2),
    N.Dense(num_units=256, activation=K.relu),
    N.Dense(num_units=len(all_labels), activation=K.softmax)
], debug=True)

K.set_training(True); y_train = f(X)
K.set_training(False); y_score = f(X)

parameters = [p for p in f.parameters if has_roles(p, [WEIGHT, BIAS])]
print("Parameters:", [p.name for p in parameters])
# ===========================================================================
# Create trainer
# ===========================================================================
trainer, hist = training.standard_trainer(
    train_data=train, valid_data=valid, test_data=test,
    X=X, y_train=y_train, y_score=y_score, y_target=y,
    parameters=parameters,
    optimizer=K.optimizers.RMSProp(lr=0.0001),
    cost_train=K.categorical_crossentropy,
    cost_score=[K.categorical_crossentropy, K.categorical_accuracy],
    confusion_matrix=len(all_labels),
    gradient_norm=True,
    batch_size=len(all_labels),
    nb_epoch=12, valid_freq=1., earlystop=5,
    # stop_callback= lambda: print("\nSTOP !!!!!!"),
    # save_callback= lambda: print("\n!!!!!! SAVE"),
    save_path=get_modelpath("commands.ai", override=True),
    save_obj=f
)
trainer.run()
