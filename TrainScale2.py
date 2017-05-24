from keras import applications
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dropout, Conv2D, Reshape, Dense, Flatten, AvgPool2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import plot_model
from keras.utils.layer_utils import print_summary
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_resnet

import keras.backend.tensorflow_backend as K
from sklearn.metrics import r2_score

import sealiondata

from time import gmtime, strftime

import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt

from DatasetScale import ScaleDataset
from kir_utils import kir_save_history

build = 35

# === Train a small conv net on top of vgg16 to detect lions' scale ===

batch_size = 16
epochs1    = 40
epochs2    = 120

resumeFrom=None
# resumeFrom='cp/31checkpoint-26-0.19.hdf5'  # <= this will trigger resuming
# resumeEpochFrom=40                   # the last trained. Will resume from resumeEpochFrom+1
# resumeEpochs=100

scale_dir = "../Sealion/TrainScale/"

img_width, img_height = 224, 224        # we also set this in ScaleDataset !!!

dataset = ScaleDataset(sealiondata.SeaLionData(), preprocess_input=preprocess_input_resnet, use_categorical=False)
times = strftime("%Y%m%d-%H-%M-%S", gmtime())

# # Test generator
# gen = dataset.generate(batch_size=batch_size)
# for j in range(5):
#     a, b = next(gen)
#     print(a.ndim, b.ndim)


# if not os.path.isdir(out_dir):
#     os.mkdir(out_dir)

# Save the model according to the conditions
checkpoint = ModelCheckpoint(filepath='cp/{}'.format(build)+'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5',
                             monitor='val_acc',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

tensorboard = TensorBoard(log_dir='./logs/scale-{}-{}'.format( build,times),
                          histogram_freq=5,
                          write_graph=False,
                          write_images=True)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


def r2_metrics(y_true, y_pred):
	ssr = K.mean((y_pred - y_true)**2, axis=0)
	sst = K.mean((y_pred - K.mean(K.mean(y_true)))**2,  axis=0)
	return 1 - K.mean(ssr/sst , axis=-1)


# ========= Model ==========

# model_pretrained = applications.VGG16(weights ="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
# print('VGG Model loaded.')

model_pretrained = applications.ResNet50(weights ="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
print('ResNet50 Model loaded.')

pretr_layer_outputs = {}
for layer in model_pretrained.layers:
    # layer.trainable = False
    pretr_layer_outputs[layer.name] = layer.get_output_at(0)

# freeze training for a few bottom layers
for layer in model_pretrained.layers[:39]:
    layer.trainable = False
    # if layer.name == 'activation_10':
    #     break

if model_pretrained.name == 'resnet50':
    x = pretr_layer_outputs['activation_49']       # we need to skip the very last layer in case of ResNet
else:
    x = model_pretrained.output

# model_pretrained.summary()

x = Conv2D(256, (7, 7), activation='elu', padding='valid', name='Kir_0')(x)

# x = AvgPool2D(pool_size=(7, 7), padding='same', name='Kir_Pool0')(x)
# x = Conv2D(2048, (1, 1), activation='elu', padding='valid', name='Kir_0')(x)

# x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv2D(1024, (1, 1), activation='elu', padding='valid', name='Kir_2')(x)
x = BatchNormalization()(x)
x = Conv2D(1, (1, 1), activation='relu', padding='valid', name='Kir_3')(x)
predictions = Reshape(target_shape=(1,))(x)

model = Model(model_pretrained.input, predictions)

if resumeFrom is None:
    model.compile(loss='mse',
                  optimizer=optimizers.Nadam(lr=0.000005),
                  metrics=[r2_metrics])
  				  # или исп  r2_score(y_true, y_pred, multioutput='variance_weighted')

    print_summary(model)
    plot_model(model, to_file='model.pdf', show_shapes=True)

    history = model.fit_generator(
        dataset.generate(batch_size=batch_size, is_training=True),
        steps_per_epoch=200,
        epochs=epochs1,
        validation_data=dataset.generate(batch_size=batch_size, is_training=False),
        validation_steps=len(dataset.test_items),
        initial_epoch=0,
        pickle_safe=True,
        verbose=1,
        callbacks=[checkpoint, tensorboard])
else:
    model.load_weights(resumeFrom)
    model.compile(loss='mean_absolute_error',        # mean_squared_error
                      optimizer=optimizers.Nadam(lr=0.000005),
                      metrics=['accuracy'],
                      decay=0.0005)

    print_summary(model)
    plot_model(model, to_file='model.pdf', show_shapes=True)

    history = model.fit_generator(
        dataset.generate(batch_size=batch_size, is_training=True),
        steps_per_epoch=200,
        epochs=epochs2,
        initial_epoch=epochs1,
        validation_data=dataset.generate(batch_size=batch_size, is_training=False),
        validation_steps=len(dataset.test_items),
        pickle_safe=True,
        verbose=1,
        callbacks=[checkpoint, tensorboard])

kir_save_history(history, 'scale_pre')
