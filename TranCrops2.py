from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dropout, Conv2D, Reshape, MaxPool2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import plot_model
from keras.utils.layer_utils import print_summary
from keras.layers.normalization import BatchNormalization

import matplotlib.pyplot as plt
from time import gmtime, strftime
import os

# === Train a small conv net on top of vgg16 ===
from kir_utils import kir_save_history

build = 11

batch_size = 8
epochs1    = 60
epochs2    = 120

# resumeFrom=None
resumeFrom='cp/crops-29-0.72.hdf5'  # <= this will trigger resuming

inp_dir = '../Sealion/'
classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]

train_data_dir = "train.224"


img_width, img_height = 224, 224

# Initiate the train and test generators
train_datagen = ImageDataGenerator(
    validation_pct=10,
    # rescale=1./128-1,
    zoom_range=0.6,
    rotation_range=30.,         # degrees
    horizontal_flip=True,
    vertical_flip=True
)

# valid_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(
    # rescale=1./128-1
)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        category='training',
        target_size=(img_width, img_height),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
validation_generator=train_datagen.flow_from_directory(
        train_data_dir,        # we split out validation set automatically: https://github.com/fchollet/keras/pull/6152
        target_size=(img_width, img_height),
        batch_size=batch_size,
        category='validation',
        class_mode='categorical')

times = strftime("%Y%m%d-%H-%M-%S", gmtime())

# Save the model according to the conditions
checkpoint = ModelCheckpoint(filepath='cp/10-crops-{epoch:02d}-{val_loss:.2f}.hdf5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

tensorboard = TensorBoard(log_dir='./logs/crops-{}'.format(times),
                          histogram_freq=5,
                          write_graph=False,
                          write_images=True)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


if resumeFrom == None:

    model_pretrained = applications.ResNet50(weights="imagenet", include_top=False,
                                             input_shape=(img_width, img_height, 3))
    print('ResNet50 Model loaded.')

    pretr_layer_outputs = {}
    for layer in model_pretrained.layers:
        # layer.trainable = False
        pretr_layer_outputs[layer.name] = layer.get_output_at(0)

    # for layer in model_pretrained.layers[:39]:
    #     layer.trainable = False
    #     # if layer.name == 'activation_10':
    #     #     break

    if model_pretrained.name == 'resnet50':
        x = pretr_layer_outputs['activation_49']  # we need to skip the very last layer in case of ResNet if 224x224
    else:
        x = model_pretrained.output

    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Conv2D(2048, (3, 3), strides=(2,2), activation='elu', padding='valid', name='Kir_01')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Conv2D(1024, (3, 3), activation='elu', padding='valid', name='Kir_02')(x)
    # x = MaxPool2D(pool_size=(7, 7), padding='same', name='Kir_Pool2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Conv2D(512, (1, 1), activation='elu', padding='valid', name='Kir_10')(x)
    x = BatchNormalization()(x)
    x = Conv2D(6, (1, 1), activation='sigmoid', padding='valid', name='Kir_30')(x)   # возможно это полезно довести 3х3 до конца, чтобы задавить соседних котиков
    predictions = Reshape(target_shape=(6,))(x)

    model = Model(input=model_pretrained.input, output=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Nadam(lr=0.0001),
                  metrics=['accuracy'])

    print_summary(model)
    plot_model(model, to_file='model.pdf', show_shapes=True)

    # ===== first stage
    history = model.fit_generator(
            train_generator,
            steps_per_epoch=300,
            epochs=epochs1,
            validation_data=validation_generator,
            validation_steps=30,
            initial_epoch=0,
            pickle_safe=True,
            verbose=1,
            callbacks=[checkpoint, tensorboard])

    kir_save_history(history, 'scale_pre')

else:
    print('Resume training after %s\'s epoch' % epochs1)
    model = load_model(resumeFrom)

    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Nadam(lr=0.00001),
                  metrics=['accuracy'],
                  decay=0.0005)

    history = model.fit_generator(
            train_generator,
            steps_per_epoch=300,
            epochs=epochs2,
            validation_data=validation_generator,
            validation_steps=30,
            workers=2,
            initial_epoch=epochs1,
            pickle_safe=True,
            verbose=1,
            callbacks=[checkpoint, tensorboard])

    kir_save_history(history, 'scale_pre')
