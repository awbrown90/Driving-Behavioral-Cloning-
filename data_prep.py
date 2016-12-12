import csv
import cv2
from scipy import ndimage
import os, os.path
import numpy as np
from six.moves import cPickle as pickle

angles = []
with open('driving_log_combine.csv', 'r') as csvfile:
	myreader = csv.reader(csvfile)
	for row in myreader:
		angles.append(row[3])

print (len(angles))
center_angles = [float(i) for i in angles]
left_angles = [(float(i)+.1*abs(float(i))+.05) for i in angles]
right_angles = [(float(i)-.1*abs(float(i))-.05) for i in angles]

left_angles = np.maximum(left_angles,-1.0)
left_angles = np.minimum(left_angles, 1.0)

right_angles = np.maximum(right_angles,-1.0)
right_angles = np.minimum(right_angles, 1.0)

angles = np.concatenate([center_angles,left_angles,right_angles])
print(len(angles))

#myfile = open('my_angles.csv','w')
#wr = csv.writer(myfile)
#wr.writerow(angles)

def load_images(num_images):
	
	filenames = os.listdir('IMG_combine')

	dataset = np.ndarray(shape=(num_images, 40, 80, 3),
                         	    dtype=np.float32)

	for image in range(num_images):
		image_file = 'IMG_combine/'+filenames[image]
		#if image is 0:
		#	print(filenames[image])

		#process_image = cv2.cvtColor(ndimage.imread(image_file), cv2.COLOR_BGR2GRAY)
		process_image = ndimage.imread(image_file)
		dataset[image,:,:,:] = cv2.resize(process_image,(80,40))

	print('Full dataset tensor:' , dataset.shape)
	print('Mean:', np.mean(dataset))
	print('Max:', np.amax(dataset))
	print('Min:', np.amin(dataset))
	print('Standard deviation:', np.std(dataset))	 
	return dataset

examples = len(angles)

def maybe_pickle(datafolder, force=True):
	set_filename = datafolder+'.p'
	if os.path.exists(set_filename) and not force:
		print('%s already present - Skipping pickling.' % set_filename)
	else:
		print('Pickling %s.' % set_filename)
		dataset = load_images(examples)
		try:
			with open(set_filename, 'wb') as f:
				pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save data to', set_filename, ':', e)
	return set_filename

def merge_datasets(pickle_file, input_angles):
	angles_set = np.ndarray(examples, dtype=np.float32)
	try:
		with open(pickle_file, 'rb') as f:
			image_set = pickle.load(f)
			angles_set = input_angles
	except Exception as e:
		print('Unable to process data from', pickle_file, ':', e)
		raise

	try:
		f = open(pickle_file, 'wb')
		save = {
    			'image_set': image_set,
    			'angles_set': angles_set,
    		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
		f.close()
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise

pickle_name = 'train'
my_dataset = maybe_pickle(pickle_name)
merge_datasets(my_dataset, angles)
