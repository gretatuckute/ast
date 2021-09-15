import pickle
import numpy as np
import warnings
from pathlib import Path
import os
import matplotlib.pyplot as plt


class SaveOutput:
	def __init__(self, avg_type='avg', rand_netw=False):
		self.outputs = []
		self.activations = {}  # create a dict with module name
		self.detached_activations = None
		self.avg_type = avg_type
		self.rand_netw = rand_netw
	
	def __call__(self, module, module_in, module_out):
		"""
		Module in has the input tensor, module out in after the layer of interest
		"""
		self.outputs.append(module_out)
		
		layer_name = self.define_layer_names(module)
		self.activations[layer_name] = module_out
	
	def define_layer_names(self, module):
		"""Define the layer name (key) for the dictionary for storing activations.
		If a layer name has already occurred, count occurrences, and append a number."""
		
		layer_name = str(module)
		current_layer_names = list(self.activations.keys())
		
		split_layer_names = [l.split('--') for l in current_layer_names]
		
		num_occurences = 0
		for s in split_layer_names:
			s = s[0]  # base name
			
			if layer_name == s:
				num_occurences += 1
		
		layer_name = str(module) + f'--{num_occurences}'
		
		if layer_name in self.activations:
			warnings.warn('Layer name already exists')
		
		return layer_name
	
	def clear(self):
		self.outputs = []
		self.activations = {}
	
	def get_existing_layer_names(self):
		for k in self.activations.keys():
			print(k)
		
		return list(self.activations.keys())
	
	def return_outputs(self):
		self.outputs.detach().numpy()
	
	def detach_one_activation(self, layer_name):
		return self.activations[layer_name].detach().numpy()
	
	def detach_activations(self):
		"""
		Detach activations (from tensors to numpy)

		Arguments:

		Returns:
			detached_activations = for each layer, the flattened activations
			packaged_data = for LSTM layers, the packaged data
		"""
		detached_activations = {}
	
		for k, v in self.activations.items():
			# print(f'Shape {k}: {v.detach().numpy().shape}')
			# print(f'Detaching activation for layer: {k}')
			activations = v.detach().numpy()
			
			if k.startswith('Conv2d'):
				# perform same flattening and reshaping as in the patch_embed layer
				# collapse across the patch and time dim (12 and 101) and squeeze
				flatten_actv = v.flatten(2).transpose(1, 2).squeeze().detach().numpy()
				# flatten actv now has [time; emb size]
				
				actv_avg = np.mean(flatten_actv, axis=0)

				detached_activations[k] = actv_avg

			if k.startswith('Linear'):
				if k.startswith('Linear(in_features=3072, out_features=768, bias=True)'): # only extract from the last part of the transformer block
					detached_activations[k] = np.mean(activations, axis=1).squeeze()
		
		self.detached_activations = detached_activations
		
		return detached_activations
	
	def store_activations(self, RESULTDIR, identifier):
		RESULTDIR = (Path(RESULTDIR))
		
		if not (Path(RESULTDIR)).exists():
			os.makedirs((Path(RESULTDIR)))
		
		if self.rand_netw:
			filename = os.path.join(RESULTDIR, f'{identifier}_activations_randnetw.pkl')
		else:
			filename = os.path.join(RESULTDIR, f'{identifier}_activations.pkl')
		
		with open(filename, 'wb') as f:
			pickle.dump(self.detached_activations, f)