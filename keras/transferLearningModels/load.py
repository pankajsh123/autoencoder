import os
import numpy as np
from PIL import Image
from resizeimage import resizeimage

def loadImage(folder,shape,mode):

	data = []
	for d in os.listdir(folder):
		img = Image.open(os.path.join(os.sep,folder,d)).convert(mode)
		if mode = 'L':
			img = np.expand_dims(img,axis = 2)
		img = resizeimage.resize_cover(img,shape)
		img = np.array(img)
		data.append(img)
	
	arr = np.array(data)
	return arr


