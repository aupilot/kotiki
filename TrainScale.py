from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dropout, Conv2D, Reshape, Dense, Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import plot_model
from keras.utils.layer_utils import print_summary
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool2D, AvgPool2D
from keras import regularizers
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input

from time import gmtime, strftime

import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt

from kir_utils import kir_save_history

# === Train a small conv net on top of vgg16 to detect lions' scale ===

batch_size = 16
epochs1    = 40
epochs2    = 80

resumeFrom=None
#resumeFrom='cp/crops-58-0.44.hdf5'  # <= this will trigger resuming
resumeEpochFrom=80                   # the last trained. Will resume from resumeEpochFrom+1
resumeEpochs=120

img_dir   = "train.scale.224"
val_dir   = "train.scale.224.val"
scale_dir = "../Sealion/TrainScale/"

# img_width, img_height = 112, 112
img_width, img_height = 224, 224


def kir_train_generator(img_dir, scale_dir, batch_size=16):

    file_names = [f for f in os.listdir(img_dir) if re.match(r'[0-9]+.*\.jpg', f)]
    file_names = sorted(file_names, key=lambda
        item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))

    random.shuffle(file_names)

    # csv = os.path.join(img_dir, 'coords.csv')
    # all_coords = np.genfromtxt(coord_file, delimiter=',', skip_header=1)    # pic,class,row,col

    # read all scales
    scale_names = [f for f in os.listdir(scale_dir) if re.match(r'[0-9]+.*\.txt', f)]
    scale_names = sorted(scale_names, key=lambda
        item: (int(item.partition('_')[0]) if item[0].isdigit() else float('inf'), item))

    scales = np.ones((1000,1))
    # scales = np.ones((len(file_names),1))
    for scale_name in scale_names:
        # iii = int(os.path.splitext(scale_name)[0])
        iii = int(scale_name.partition('_')[0])
        scales[iii] = np.loadtxt(os.path.join(scale_dir, scale_name))

    batchX = np.zeros((batch_size, img_width, img_height, 3))
    batchY = np.zeros((batch_size, 1))
    ptr = 0

    while True:
        for filename in file_names:
            # get file number
            iii = int(filename.partition('_')[0])

            img = load_img(os.path.join(img_dir,filename))
            xx = img_to_array(img)
            # xx = xx/128.-1.

            batchX[ptr,:,:,:] = xx
            batchY[ptr] = scales[iii]

            ptr = ptr + 1

            if ptr == batch_size:
                # output
                # yield batchX, np.expand_dims(batchY, axis=0)
                yield preprocess_input(batchX), batchY
                # yield np.zeros((16,224,224,3)), np.zeros((16,1))
                ptr = 0
                batchY = np.zeros((batch_size, 1))

# # Test generator
# gen = kir_train_generator(img_dir=img_dir, scale_dir=scale_dir, batch_size=batch_size)
# for j in range(15000):
#     a, b = next(gen)
# print(a.ndim, b.ndim)

times = strftime("%Y%m%d-%H-%M-%S", gmtime())

# if not os.path.isdir(out_dir):
#     os.mkdir(out_dir)

# Save the model according to the conditions
checkpoint = ModelCheckpoint(filepath='cp/scale-{epoch:02d}-{val_loss:.2f}.hdf5',
                             monitor='val_acc',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)
tensorboard = TensorBoard(log_dir='./logs/scale-{}'.format(times),
                          histogram_freq=5,
                          write_graph=False,
                          write_images=True)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# ========= Model ==========
if resumeFrom == None:

    # model_pretrained = applications.VGG16(weights ="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
    # print('VGG Model loaded.')

    model_pretrained = applications.ResNet50(weights ="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
    print('ResNet50 Model loaded.')

    pretr_layer_outputs = {}
    for layer in model_pretrained.layers:
        layer.trainable = False
        pretr_layer_outputs[layer.name] = layer.get_output_at(0)

    # freeze training for a few bottom layers
    # for layer in model_pretrained.layers[:39]:
    #     layer.trainable = False
        # if layer.name == 'activation_10':
        #     break

    if model_pretrained.name == 'resnet50':
        x = pretr_layer_outputs['activation_49']       # we need to skip the very last layer in case of ResNet
    else:
        x = model_pretrained.output

    # model_pretrained.summary()

    x = Conv2D(1024, (7, 7), activation='elu', padding='valid', name='Kir_0')(x)
    # x = AvgPool2D(pool_size=(7, 7), padding='same', name='Kir_0')(x)
    # x = MaxPool2D(pool_size=(7, 7), padding='same', name='Kir_0')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='elu', padding='valid', name='Kir_2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, (1, 1), activation='relu', padding='valid', name='Kir_3')(x)
    predictions = Reshape(target_shape=(1,))(x)

    # x = Flatten()(x)
    # x = Dropout(0.2)(x)
    # x = BatchNormalization()(x)
    # x = Dense(512, activation='elu', name='Kir_FC1')(x)
    # x = BatchNormalization()(x)
    # x = Dense(256, activation='elu', name='Kir_FC2')(x)
    # predictions = Dense(1, activation='linear',
    #                     kernel_regularizer=regularizers.l2(0.01),
    #                     activity_regularizer=regularizers.l1(0.01),
    #                     name='Kir_FC3')(x)

    model = Model(model_pretrained.input, predictions)
    # model = Model(input=model_pretrained.input, output=predictions)

    # model.compile(loss = "mean_absolute_error",
    #               optimizer = optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True),
    #               metrics=["accuracy"],
    #               decay=0.0005)

    # model.compile(loss = "mean_squared_error",
    #               optimizer = optimizers.SGD(lr=0.00001, momentum=0.9, nesterov=True),
    #               metrics=["accuracy"],
    #               decay=0.0005)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Nadam(lr=0.0001),
                  metrics=['accuracy'])

    print_summary(model)
    plot_model(model, to_file='model.pdf', show_shapes=True)

    # ===== first stage
    history = model.fit_generator(
        kir_train_generator(img_dir=img_dir, scale_dir=scale_dir, batch_size=batch_size),
        steps_per_epoch=200,
        epochs=epochs1,
        validation_data=kir_train_generator(img_dir=val_dir, scale_dir=scale_dir, batch_size=batch_size),
        validation_steps=20,
        initial_epoch=0,
        pickle_safe=True,
        verbose=1,
        callbacks=[checkpoint, tensorboard])

    kir_save_history(history, 'scale_pre')

    # ==== second stage
    # Unfreeze the first 6 layers
    # for layer in model.layers[:7]:
    #     layer.trainable = True
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizers.Nadam(lr=0.000001),
    #               metrics=['accuracy'])

    # history = model.fit_generator(
    #         train_generator,
    #         steps_per_epoch=1000,
    #         epochs=epochs2,
    #         validation_data=validation_generator,
    #         validation_steps=100,
    #         workers=2,
    #         initial_epoch=epochs1,
    #         pickle_safe=True,
    #         verbose=2,
    #         callbacks=[checkpoint, tensorboard])
    #
    # save_history(history, 'post')
else:
    print('Resume training after %s\'s epoch' % resumeEpochFrom)
    model = load_model(resumeFrom)

    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    model.compile(loss='mean_absolute_error',
                  optimizer=optimizers.Nadam(lr=0.000002),
                  metrics=['accuracy'])

    # Continue training
    history = model.fit_generator(
        kir_train_generator(img_dir=img_dir, scale_dir=scale_dir, batch_size=batch_size),
        steps_per_epoch=200,
        epochs=resumeEpochs,
        validation_data=kir_train_generator(img_dir=val_dir, scale_dir=scale_dir, batch_size=batch_size),
        validation_steps=20,
        initial_epoch=resumeEpochFrom,
        pickle_safe=True,
        verbose=1,
        callbacks=[checkpoint, tensorboard])

    kir_save_history(history, 'scale_pre')
