from tqdm import tqdm
import os
import torch
import numpy as np
from models import ASTModel
import torchaudio
import matplotlib.pyplot as plt

# import extractor hook functions
from extractor_utils import SaveOutput

## SETTINGS ##
RESULTDIR = '/Users/gt/Documents/GitHub/aud-dnn/aud_dnn/model-actv/AST/'
DATADIR = '/Users/gt/Documents/GitHub/aud-dnn/data/stimuli/165_natural_sounds_16kHz/'

run_only_missing_files = True

files = [f for f in os.listdir(DATADIR) if os.path.isfile(os.path.join(DATADIR, f))]
wav_files = [f for f in files if f.endswith('wav')]

if run_only_missing_files:
	# if only running remaining files:
	# Get identifier (sound file name)
	identifiers = [f.split('/')[-1].split('.')[0] for f in wav_files]
	identifier_pkls = [f'{f}_activations.pkl' for f in identifiers]
	existing_actv = [f for f in os.listdir(RESULTDIR) if os.path.isfile(os.path.join(RESULTDIR, f))]
	set_files = set(identifier_pkls) - set(existing_actv)
	wav_files = [DATADIR + f.split('_activations')[0] + '.wav' for f in set_files]

def make_features(wav_name, mel_bins, target_length=1024):
	"""Copied over from inference.py
	Looks to be based on the audioset model given the normalization -4 and 4"""
	waveform, sr = torchaudio.load(wav_name)
	
	fbank = torchaudio.compliance.kaldi.fbank(
		waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
		window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
		frame_shift=10)
	
	n_frames = fbank.shape[0]
	
	p = target_length - n_frames # check for target length, for the audioset trained model it is 1024 frames
	if p > 0:
		m = torch.nn.ZeroPad2d((0, 0, 0, p))
		fbank = m(fbank)
	elif p < 0:
		fbank = fbank[0:target_length, :]
	
	fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
	
	# plt.imshow(fbank, interpolation=None)
	# plt.show()
	
	return fbank

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'

# assume each input spectrogram has 100 time frames
# assume the task has 527 classes
label_dim = 527
input_tdim = 1024  # audioset default

# Load the pretrained AST model
model = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=True, audioset_pretrain=True)

### LOOP OVER AUDIO FILES ###
for filename in tqdm(wav_files):
	
	input = make_features(filename, mel_bins=128, target_length=1024)
	# add a batch dim
	input = input[None, :, :]

	# put model in eval mode:
	model.eval()
	
	# Write hooks for the model
	save_output = SaveOutput()
	
	hook_handles = []
	layer_names = []
	for idx, layer in enumerate(model.modules()):
		layer_names.append(layer)
		# print(layer)
		if isinstance(layer, torch.nn.modules.conv.Conv2d): # this is the "single" patch embed layer
			# print('Fetching conv handles!\n')
			handle = layer.register_forward_hook(save_output)
			hook_handles.append(handle)
		if type(layer) == torch.nn.modules.Linear:
			# print('Fetching Linear handles!\n')
			handle = layer.register_forward_hook(save_output)
			hook_handles.append(handle)
	
	
	output = model(input)
	# output should be in shape [1, 527], i.e., 10 samples, each with prediction of 527 classes.
	# print(output.shape)
	
	detached_activations = save_output.detach_activations()
	
	# Add the output features
	detached_activations['Final'] = output.detach().numpy() # this is the same as the very last linear layer, ie. Linear in=768 and out=527
	
	# plt.plot(detached_activations['Linear(in_features=3072, out_features=768, bias=True)--7'])
	# plt.show()
	
	# Store and save activations
	# Get identifier (sound file name)
	id1 = filename.split('/')[-1]
	identifier = id1.split('.')[0]
	
	save_output.store_activations(RESULTDIR=RESULTDIR, identifier=identifier)
