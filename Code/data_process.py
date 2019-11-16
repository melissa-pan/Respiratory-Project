from pydub import AudioSegment

import shutil
import pandas as pd
import numpy as np
import os

#path = '.'
# Get all patient's diagnositic information
# ------------------------------------------------------------------------------------
diag_path = '../kaggle_db/Respiratory_Sound_Database/patient_diagnosis.csv'
diag = pd.read_csv(diag_path, header=None, index_col=False, names=['patient_id', 'diagnosis'])

# Total number of class
diseases = set(diag["diagnosis"])

dict_data = {}

for d in diseases:
	dict_data[d] = []

# Create new directory for processed data
# ------------------------------------------------------------------------------------
new_path = '../kaggle_db/processed_sound'
data_file = new_path + '/data.csv'

def createClassFolders(path, classes):
	for c in classes:
		if not os.path.exists(path + '/' + c):
			os.makedirs(path + '/' + c)

if not os.path.exists(new_path):
	os.makedirs(new_path)
	os.makedirs(new_path + '/train')
	os.makedirs(new_path + '/validation')
	os.makedirs(new_path + '/test')
	createClassFolders(new_path + '/train', diseases)
	createClassFolders(new_path + '/validation', diseases)
	createClassFolders(new_path + '/test', diseases)
else:
	print("Directory already exists")

# Read and parse audio files
# ------------------------------------------------------------------------------------
db_path = '../kaggle_db/Respiratory_Sound_Database'
txt_files = [f for f in os.listdir(db_path + "/txt") if f.endswith('.txt')]
wav_files = [f for f in os.listdir(db_path + "/wav") if f.endswith('.wav')]
txt_cols = ['Start', 'End', 'Crackles', 'Wheezes']

filenames = []
labels = []

if len(txt_files) != len(wav_files):
	raise Error("Warning! number of text files and wave files does not match")

counter = 0
for tfile in txt_files:
	print("file: {}".format(tfile))
	filename, extension = os.path.splitext(tfile)

	tf = db_path + '/txt/' + tfile
	wf = db_path + '/wav/' + filename + '.wav'

	# Find corresponding label
	tf_split = tfile.split("_")
	patient_id = int(tf_split[0])

	info = diag.loc[diag['patient_id'] == patient_id]
	label= info['diagnosis'].to_string(index=False).strip()

	# Read time cycle from text file
	cycles = pd.read_csv(tf, sep="\t", header=None, names=txt_cols)
	# Read sound file
	wav = AudioSegment.from_wav(wf)

	for i, c in cycles.iterrows():
		# Get cycle time and convert to millisecond
		st = int(float(c['Start']) * 1000)
		et = int(float(c['End']) * 1000)

		print("time length: {}".format(et - st))

		audio = wav[st:et]
		fname = label + "_" + str(patient_id) + "_" + str(counter) + '.wav'
		audio.export(new_path + '/' + fname, format="wav")

		# filenames.append(fname)
		# labels.append(label)

		dict_data[label].append(fname)

		counter += 1

np_data = np.column_stack((filenames, labels))
np.random.seed(360)
np.random.shuffle(np_data)
#df_data = pd.DataFrame(list(zip(filenames, labels)), columns =['filename', 'label'])

#print(df_data['label'].value_counts())

# Split into train validation and test sets
# ------------------------------------------------------------------------------------
def split_and_save(np_data):
	trIdx = int(0.5*len(np_data))
	vlIdx = int(0.2*len(np_data))
	train = np_data[0: trIdx]
	valid = np_data[trIdx: trIdx + vlIdx]
	test  = np_data[trIdx + vlIdx:,]

	print("Train data: {}".format(len(train)))
	print("Valid data: {}".format(len(valid)))
	print("Test  data: {}".format(len(test)))

	# Move data into corresponding folders
	# ------------------------------------------------------------------------------------
	for fname, label in train:
		try:
			shutil.move(new_path + '/' + fname, new_path + '/train/' + label + '/' + fname)
		except Exception as e:
			print("Warning: {} has error".format(fname))
			pass
		

	for fname, label in valid:
		try:
			shutil.move(new_path + '/' + fname, new_path + '/validation/' + label + '/' + fname)
		except Exception as e:
			print("Warning: {} has error".format(fname))
			pass

	for fname, label in test:
		try:
			shutil.move(new_path + '/' + fname, new_path + '/test/' + label + '/' + fname)
		except Exception as e:
			print("Warning: {} has error".format(fname))
			pass

# Organize data by labels
# ------------------------------------------------------------------------------------
for k,v in dict_data.items():
	np_label = np.full((len(v)), k)
	np_data = np.column_stack((v, np_label))
	print("label: {}".format(k))
	split_and_save(np_data)


