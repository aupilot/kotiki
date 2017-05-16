from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dropout, Conv2D, Reshape
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import plot_model
from keras.utils.layer_utils import print_summary
from keras.layers.normalization import BatchNormalization

import matplotlib.pyplot as plt
from time import gmtime, strftime
import os

# === Train a small conv net on top of vgg16 ===

batch_size = 16
epochs1    = 40
epochs2    = 80

resumeFrom=None
#resumeFrom='cp/crops-58-0.44.hdf5'  # <= this will trigger resuming
resumeEpochFrom=80                   # the last trained. Will resume from resumeEpochFrom+1
resumeEpochs=120


def save_history(history, prefix):
    if 'acc' not in history.history:
        return

    plots_dir = "plots"

    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    img_path = os.path.join(plots_dir, '{}-%s.jpg'.format(prefix))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(img_path % 'accuracy')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(img_path % 'loss')
    plt.close()


inp_dir = '../Sealion/'
classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]

train_data_dir = "data/train.112"
# validation_data_dir = "data/val"


img_width, img_height = 112, 112
# img_width, img_height = 224, 224

# Initiate the train and test generators
train_datagen = ImageDataGenerator(
    validation_pct=10,
    # rescale=1./128-1,
    zoom_range=0.2,
    rotation_range=30.,         # degrees
    horizontal_flip=True,
    vertical_flip=True
)

# valid_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(
    # rescale=1./128-1
)

train_generator = train_datagen.flow_from_directory(
        'train.112',  # this is the target directory
        category='training',
        target_size=(img_width, img_height),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
# we use the same generator that splits automatically
#validation_generator = valid_datagen.flow_from_directory(
validation_generator=train_datagen.flow_from_directory(
        'train.112',        # we split out validation set automatically: https://github.com/fchollet/keras/pull/6152
        target_size=(img_width, img_height),
        batch_size=batch_size,
        category='validation',
        class_mode='categorical')

times = strftime("%Y%m%d-%H-%M-%S", gmtime())
# try:
#     os.stat('./logs/crops/{}'.format(times))
# except:
#     os.mkdir('./logs/crops/{}'.format(times))

# Save the model according to the conditions
# checkpoint  = ModelCheckpoint(filepath='checkpoints/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(filepath='cp/crops-{epoch:02d}-{val_loss:.2f}.hdf5',
                             monitor='val_acc',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)
tensorboard = TensorBoard(log_dir='./logs/crops-{}'.format(times),
                          histogram_freq=5,
                          write_graph=False,
                          write_images=True)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# ========= Model ==========
if resumeFrom == None:
    modelVGG = applications.VGG16(weights ="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
    print('VGG Model loaded.')

    # Freeze the first 15 layers which we don't want to train
    for layer in modelVGG.layers[:16]:
        layer.trainable = False

    x = modelVGG.output

    # x = Conv2D(512, (3, 3), activation='elu', padding='valid', name='Kir_1')(x)
    # x = Dropout(0.5)(x)
    # x = Conv2D(512, (1, 1), activation='elu', padding='valid', name='Kir_2')(x)
    # x = Conv2D(6, (1, 1), activation='sigmoid', padding='valid', name='Kir_3')(x)
    # predictions = Reshape(target_shape=(6,))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Conv2D(512, (1, 1), activation='elu', padding='valid', name='Kir_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='elu', padding='valid', name='Kir_2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(6, (3, 3), activation='sigmoid', padding='valid', name='Kir_3')(x)   # возможно это полезно довести 3х3 до конца, чтобы задавить соседних котиков
    predictions = Reshape(target_shape=(6,))(x)

    model = Model(input=modelVGG.input, output=predictions)

    # compile the model
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-08, decay=0.0005), metrics=['accuracy'])
    # model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"], decay=0.0005)

    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08),
    #               metrics=['accuracy'])

    # model.compile(loss = "categorical_crossentropy",
    #               optimizer = optimizers.SGD(lr=0.00003, momentum=0.9, nesterov=True, decay=1e-6),
    #               metrics=["accuracy"],
    #               decay=0.0005)

    # model.compile(loss = "mean_absolute_error",
    #               optimizer = optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True),
    #               metrics=["accuracy"],
    #               decay=0.0005)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Nadam(lr=0.00001),
                  metrics=['accuracy'])

    print_summary(model)
    plot_model(model, to_file='model.pdf', show_shapes=True)

    # ===== first stage
    history = model.fit_generator(
            train_generator,
            steps_per_epoch=1000,
            epochs=epochs1,
            validation_data=validation_generator,
            validation_steps=100,
            initial_epoch=0,
            pickle_safe=True,
            verbose=1,
            callbacks=[checkpoint, tensorboard])

    save_history(history, 'pre')

    # ==== second stage
    # Unfreeze the first 6 layers
    for layer in model.layers[:7]:
        layer.trainable = True

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Nadam(lr=0.000001),
                  metrics=['accuracy'])
    # model.compile(loss="categorical_crossentropy",
    #               optimizer=optimizers.SGD(lr=0.000001, momentum=0.9, nesterov=True),
    #               metrics=["accuracy"],
    #               decay=0.0005)
    #
    #  model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizers.Adadelta(lr=0.01, rho=0.95, epsilon=1e-08, decay=0.0),
    #               metrics=['accuracy'])

    history = model.fit_generator(
            train_generator,
            steps_per_epoch=1000,
            epochs=epochs2,
            validation_data=validation_generator,
            validation_steps=100,
            workers=2,
            initial_epoch=epochs1,
            pickle_safe=True,
            verbose=1,
            callbacks=[checkpoint, tensorboard])

    save_history(history, 'post')
else:
    print('Resume training after %s\'s epoch' % resumeEpochFrom)
    model = load_model(resumeFrom)

    # Unfreeze the first 5 layers
    for layer in model.layers[:16]:
        layer.trainable = True

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.SGD(lr=0.000001, momentum=0.9, nesterov=True, decay=1e-6),
                  metrics=["accuracy"],
                  decay=0.0005)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizers.Adadelta(lr=0.05, rho=0.95, epsilon=1e-08, decay=0.0),
    #               metrics=['accuracy'])

    history = model.fit_generator(
            train_generator,
            steps_per_epoch=1000,
            epochs=resumeEpochs,
            validation_data=validation_generator,
            validation_steps=50,
            workers=2,
            initial_epoch=resumeEpochFrom,
            pickle_safe=True,
            verbose=1,
            callbacks=[checkpoint, tensorboard])

    save_history(history, 'post')
