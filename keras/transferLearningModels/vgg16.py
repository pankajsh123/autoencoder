from keras import applications , optimizers
from keras.layers import Dense, Activation , core , Convolution2D, MaxPooling2D , Dropout, Flatten , normalization
from keras.utils import np_utils
from keras.models import Model

def featureExtractor(num_classes,saveas,inputData,label,testData,labelt):

	base_model = applications.VGG16(weights='imagenet', include_top=False,pooling = False,input_shape = (224, 224, 3))

	last = base_model.output
	last = Dense(1024, activation='relu')(last)
	last = Dropout(0.25)(last)
	last = Dense(256, activation='relu')(last)
	last = Dropout(0.25)(last)
	pred = Dense(num_classes, activation='softmax')(last)

	for layer in base_model.layers[:19]:
		layer.trainable = False

	model = Model(inputs = base_model.input, outputs=pred)
	model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
	model.fit(inputData,label,batch_size = 32,epochs = 10,verbose = 2,shuffle = True)
	model.evaluate(testData,labelt,verbose = 1,batch_size = 32)
	model.save(saveas + '.h5')

def fineTuning(num_classes,layerNum,saveas,inputData,label,testData,labelt):
	
	base_model = applications.VGG16(weights='imagenet', include_top=False,pooling = False,input_shape = (224, 224, 3))

	last = base_model.output
	last = Dense(1024, activation='relu')(last)
	last = Dropout(0.25)(last)
	last = Dense(256, activation='relu')(last)
	last = Dropout(0.25)(last)
	pred = Dense(num_classes, activation='softmax')(last)

	for layer in base_model.layers[:19]:
		layer.trainable = False

	model = Model(inputs = base_model.input, outputs=pred)
	model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
	model.fit(inputData,label,batch_size = 32,epochs = 10,verbose = 0,shuffle = True)

	for layer in model.layers[:layerNum]:
		layer.trainable = False
	for layer in model.layers[layerNum:]:
		layer.trainable = True
	
	model.compile(optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
	model.fit(inputData,label,batch_size = 32,epochs = 10,verbose = 2,shuffle = True)
	model.evaluate(testData,labelt,verbose = 1,batch_size = 32)
	model.save(saveas + '.h5')

def retrain(num_classes,saveas,inputData,label,testData,labelt):

	base_model = applications.VGG16(weights=None, include_top=True,input_shape = (224, 224, 3),classes = num_classes)
	model = Model(inputs = base_model.input , outputs = base_model.output)
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
	model.fit(inputData,label,batch_size = 32,epochs = 10,verbose = 2,shuffle = True)
	model.evaluate(testData,labelt,verbose = 1,batch_size = 32)
	model.save(saveas + '.h5')
	

def main(num_classes,finetuning = False,layerNum = 15,retrain = False,saveas,inputData,label,testData,labelt):
	
	if retrain == True:
		retrain(num_classes,saveas,inputData,label,testData,labelt)
	else if finetuning == True:
		fineTuning(num_classes,layerNum,saveas,inputData,label,testData,labelt)
	else 
		featureExtractor(num_classes,saveas,inputData,label,testData,labelt)

main(num_classes,finetuning,layerNum,retrain,saveas,inputData,label,testData,labelt)
	


