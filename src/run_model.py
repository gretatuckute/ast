from tqdm import tqdm
import os
import torch
import numpy as np
from models import ASTModel
import torchaudio
import matplotlib.pyplot as plt
import pickle
# import extractor hook functions
from extractor_utils import SaveOutput
import random

## SETTINGS ##
RESULTDIR = '/Users/gt/Documents/GitHub/aud-dnn/aud_dnn/model-actv/AST'
DATADIR = '/Users/gt/Documents/GitHub/aud-dnn/data/stimuli/165_natural_sounds_16kHz/'

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

get_sound_stats = False # compute mean and std of the dataset of interest (165 sounds)
run_only_missing_files = True
rand_netw = False # if True, run the model with permuted weights
sound_level_check = 10 # If not None, multiply the raw sound by this value to check model activations

# If sound_level_check is not None, append SL{sound_level_check} to the resultdir
if sound_level_check is not None:
	# If period in the sound level, drop it
	if '.' in str(sound_level_check):
		sound_level_check_str = str(sound_level_check).replace('.', '')
	else:
		sound_level_check_str = str(sound_level_check)
	RESULTDIR = RESULTDIR + f'SL{sound_level_check_str}'

if not os.path.exists(RESULTDIR):
	os.makedirs(RESULTDIR)

files = [f for f in os.listdir(DATADIR) if os.path.isfile(os.path.join(DATADIR, f))]
wav_files_identifiers = [f for f in files if f.endswith('wav')]
wav_files_paths = [DATADIR + f for f in wav_files_identifiers]

if run_only_missing_files:
	# if only running remaining files:
	# Get identifier (sound file name)
	identifiers = [f.split('/')[-1].split('.')[0] for f in wav_files_identifiers]
	if rand_netw:
		identifier_pkls = [f'{f}_activations_randnetw.pkl' for f in identifiers]
	else:
		identifier_pkls = [f'{f}_activations.pkl' for f in identifiers]
	existing_actv = [f for f in os.listdir(RESULTDIR) if os.path.isfile(os.path.join(RESULTDIR, f))]
	set_files = set(identifier_pkls) - set(existing_actv)
	wav_files_paths = [DATADIR + f.split('_activations')[0] + '.wav' for f in set_files]




def make_features(wav_name, mel_bins, target_length=1024, sound_level_check=None):
	"""Copied over from inference.py
	Looks to be based on the audioset model given the normalization -4 and 4
	If sound_level_check is not None, multiply the raw sound by this value to check model activations
	"""
	waveform, sr = torchaudio.load(wav_name)

	if sound_level_check is not None:
		# Multiply tensor by sound_level_check
		waveform = torch.mul(waveform, sound_level_check) # same as waveform2 = waveform * sound_level_check
	
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


def get_sound_stats_func(wav_name, mel_bins, target_length=1024, new_rms=None, new_rms_scale=None):
	"""Copy of make_features

	If new_rms is not None, then the sound is scaled to have the new_rms amplitude.
	If new_rms_scale is not None, then the sound is just scaled by the new_rms_scale.

	"""

	waveform, sr = torchaudio.load(wav_name)
	# rms_predemean = ch_rms(waveform)
	waveform = ch_demean(waveform)


	# print RMS
	# rms = ch_rms(waveform)
	# print(f'RMS pre demean: {rms_predemean} and post demean: {rms}')

	if new_rms is not None:
		waveform = waveform / ch_rms(waveform) * new_rms

	if new_rms_scale is not None:
		waveform = waveform * new_rms_scale

	fbank = torchaudio.compliance.kaldi.fbank(
		waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
		window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
		frame_shift=10)

	n_frames = fbank.shape[0]

	p = target_length - n_frames  # check for target length, for the audioset trained model it is 1024 frames
	if p > 0:
		m = torch.nn.ZeroPad2d((0, 0, 0, p))
		fbank = m(fbank)
	elif p < 0:
		fbank = fbank[0:target_length, :]

	# fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)

	# plt.imshow(fbank, interpolation=None)
	# plt.show()

	return fbank


def ch_demean(x, dim=-1):
	'''
    Helper function to mean-subtract tensor.

    Args
    ----
    x (tensor): tensor to be mean-subtracted
    dim (int): kwarg for torch.mean (dim along which to compute mean)

    Returns
    -------
    x_demean (tensor): mean-subtracted tensor
    '''
	x_demean = torch.sub(x, torch.mean(x, dim=dim))
	return x_demean


def ch_rms(x, dim=-1):
	'''
    Helper function to compute RMS amplitude of a tensor.

    Args
    ----
    x (tensor): tensor for which RMS amplitude should be computed
    dim (int): kwarg for torch.mean (dim along which to compute mean)

    Returns
    -------
    rms_x (tensor): root-mean-square amplitude of x
    '''
	rms_x = torch.sqrt(torch.mean(torch.pow(x, 2), dim=dim))
	return rms_x



if get_sound_stats:
	lst_fbanks = []
	for wav_file in tqdm(wav_files_paths):
		fbank = get_sound_stats_func(wav_name=wav_file,
								mel_bins=128,
								target_length=1024,
								new_rms=None,
								new_rms_scale=10) # 10 and 0.1

		lst_fbanks.append(fbank)

	# Compute the mean and std of the dataset
	mean_fbank = torch.mean(torch.stack(lst_fbanks), dim=0)
	mean_dataset = mean_fbank.mean()
	std_dataset = mean_fbank.std()


	print(f'Mean of the dataset: {mean_dataset:3f} and std of the dataset: {std_dataset:3f}')








# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'

# assume each input spectrogram has 100 time frames
# assume the task has 527 classes
label_dim = 527
input_tdim = 1024  # audioset default

# Load the pretrained AST model
model = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=True, audioset_pretrain=True)

if rand_netw:
	state_dict = model.state_dict()
	state_dict_rand = {}
	print('OBS! RANDOM NETWORK!')

	## The following code was used to generate indices for random permutation ##
	if not os.path.exists(os.path.join(os.getcwd(), 'AST_randnetw_indices.pkl')):
		print('________ Generating random indices for permuted architecture ________')
		d_rand_idx = {}  # create dict for storing the indices for random permutation
		for k, v in state_dict.items():
			w = state_dict[k]
			idx = torch.randperm(w.nelement())  # create random indices across all dimensions
			d_rand_idx[k] = idx

		with open(os.path.join(os.getcwd(), 'AST_randnetw_indices.pkl'), 'wb') as f:
			pickle.dump(d_rand_idx, f)

	else:
		print('________ Loading random indices from permuted architecture ________')
		d_rand_idx = pickle.load(open(os.path.join(os.getcwd(), 'AST_randnetw_indices.pkl'), 'rb'))
	
	for k, v in state_dict.items():
		w = state_dict[k]
		# Load random indices
		print(f'________ Loading random indices from permuted architecture for {k} ________')
		idx = d_rand_idx[k]
		rand_w = w.view(-1)[idx].view(w.size())  # permute using the stored indices, and reshape back to original shape
		state_dict_rand[k] = rand_w

	model.load_state_dict(state_dict_rand)

### LOOP OVER AUDIO FILES ###
for filename in tqdm(wav_files_paths):

	input = make_features(filename, mel_bins=128, target_length=1024, sound_level_check=sound_level_check)

	# add a batch dim
	input = input[None, :, :]

	# put model in eval mode:
	model.eval()

	# Write hooks for the model
	save_output = SaveOutput(rand_netw=rand_netw)

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
	detached_activations['Final'] = output.detach().numpy().squeeze() # this is the same as the very last linear layer, ie. Linear in=768 and out=527

	# plt.plot(detached_activations['Linear(in_features=3072, out_features=768, bias=True)--7'])
	# plt.show()

	# Store and save activations
	# Get identifier (sound file name)
	id1 = filename.split('/')[-1]
	identifier = id1.split('.')[0]

	save_output.store_activations(RESULTDIR=RESULTDIR, identifier=identifier)
