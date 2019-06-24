import pickle
import copy
import glob, os
import math
import training_convnet

def BestErrorFile(folder_name):
	
	filename_best = ''
	best_error = 10**6

	current_dir = os.getcwd()
	os.chdir(folder_name)
	for file in glob.glob("*.pkl"):
		with open(file, 'rb') as f:
			params = pickle.load(f)
			if not math.isnan(params['minRegTest']):
				if params['minRegTest'] < best_error:
					filename_best = file
					best_error = params['minRegTest']
        os.chdir(current_dir)

	return filename_best

folder_name = './Burgers_exp21dd/'
file_name = BestErrorFile(folder_name)
if file_name == '':
	print('No files have finite error')
	quit()
else:
	pkl_file = folder_name + file_name

with open(pkl_file, 'rb') as f:
     params = pickle.load(f)

params['restore'] = 1
params['auto_first'] = 0
params['max_time'] = 12*60*60
params['min_5min'] = 10**4
params['min_20min'] = 10**3
params['min_40min'] = 10**3
params['min_1hr'] = 10**2
params['min_2hr'] = 10**1
params['min_3hr'] = 10**1
params['min_4hr'] = 10**1
params['min_halfway'] = 5

params['model_restore_path'] = params['model_path']

training_convnet.main_exp(copy.deepcopy(params))
