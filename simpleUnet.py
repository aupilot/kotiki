import os
import numpy as np
import re
import cv2

from time import gmtime, strftime
from unet import UNet
from kir_utils import kir_save_history

from keras.preprocessing.image import load_img, img_to_array
from keras import optimizers
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import plot_model
from keras.utils.layer_utils import print_summary

import matplotlib.pyplot as plt

np.random.seed(1337)

resumeFrom=None
#resumeFrom='checkpoints/checkpoint-33-0.44.hdf5'  # <= this will trigger resuming
resumeEpochFrom=40                                # the last trained. Will resume from resumeEpochFrom+1
resumeEpochs=100

train_dir = '../Sealion/'
classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]

batch_size = 4
epochs1    = 20
epochs2    = 40

# pad_width, pad_height   = 120, 120              # size for padding of tales надо понять насколько они должны быть большие!
# tale_width, tale_height = 750, 750              # size of tales before padding
# img_width, img_height   = tale_width + pad_width, tale_height + pad_height
# out_width, out_height   = 200, 200

pad_width, pad_height   = 50, 50              # size for padding of tales надо понять насколько они должны быть большие!
tale_width, tale_height = 500, 500              # size of tales before padding
img_width, img_height   = tale_width + pad_width, tale_height + pad_height
out_width, out_height   = 120, 120

# ====== generators =====
# returns X, Y with
# X:  [batch, width, height, color]
# Y:  [batch, width, height, class]

def kir_generator(img_dir, coord_file='',  b_size=18):
    # the dir must contain 'coords.csv'

    global pad_width, pad_height
    global tale_width, tale_height
    global img_width, img_height
    global out_width, out_height

    half_w = int(pad_width  / 2)
    half_h = int(pad_height / 2)

    file_names = [f for f in os.listdir(img_dir) if re.match(r'[0-9]+.*\.jpg', f)]
    file_names = sorted(file_names, key=lambda
        item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))

    # csv = os.path.join(img_dir, 'coords.csv')
    all_coords = np.genfromtxt(coord_file, delimiter=',', skip_header=1)    # pic,class,row,col

    batchX = np.zeros((b_size, img_width, img_height, 3))
    batchY = np.zeros((b_size, out_width, out_height, 5))

    coords_scale = out_width / tale_width

    ptr = 0

    while True:

        for filename in file_names:
            # load a test sample
            img = load_img(img_dir+'/'+filename)
            width, height = img.size
            img_resized = img.resize((4500, 3000))
            k = 3000./float(height)

            # get file number
            iii = int(os.path.splitext(filename)[0])
            img_coords = all_coords[all_coords[:,0] == iii]     # get coords for this image
            img_coords[:,2:4] = img_coords[:,2:4] * k           # scale coords

            # special type image to np array
            x = img_to_array(img_resized)
            x = x/128.-1

            # tale the picture to 6 pcs
            for cy in range(0, int(3000/tale_height)):
                for cx in range(0, int(4500/tale_width)):
                    # make a padded image from crop
                    xx = x[cy * tale_height:(cy + 1) * tale_height, cx * tale_width:(cx + 1) * tale_width, :]
                    # now we add borders around the tale
#                    xx_padded = cv2.copyMakeBorder(xx, half_h, half_h, half_w, half_w, cv2.BORDER_REFLECT_101)  #top, bottom, left, right - pad em all!
                    xx_padded = np.pad(xx, ((half_h,half_h), (half_w, half_w), (0,0)), mode='edge')
                    batchX[ptr, :,:,: ] = xx_padded

                    # choose coords that belogs to the tale
                    specific_coords = img_coords[(img_coords[:, 3] >= (cx * tale_width))    # x
                                            & (img_coords[:, 3] < ((cx + 1) * tale_width))
                                            & (img_coords[:, 2] >= (cy * tale_height))      # y
                                            & (img_coords[:, 2] < ((cy + 1) * tale_height))
                                                ]

                    # Adjust coords to the tale begining
                    specific_coords[:,2] = specific_coords[:,2] - cy * tale_height
                    specific_coords[:,3] = specific_coords[:,3] - cx * tale_width

                    # make an output map
                    # outMap = np.zeros((out_width, out_height, 5), dtype=int)
                    for s in range(5):
                        # find all lions' coords
                        lion_coords = specific_coords[specific_coords[:,1] == s]

                        # fill the map. Perhaps change just dots to Gaussians with R>1
                        for i in range(np.size(lion_coords, 0)):
                            sr = int(lion_coords[i,2]*coords_scale)
                            sc = int(lion_coords[i,3]*coords_scale)
                            batchY[ptr, sr-1:sr+2, sc-1:sc+2, s] = 1.
                            # print(sr, sc)

                    ptr = ptr+1
                    if ptr == b_size:
                        # output
                        yield batchX, batchY
                        ptr = 0
                        batchY = np.zeros((b_size, out_width, out_height, 5))


# kir_generator('../Sealion', 6)


times=strftime("%Y%m%d-%H-%M-%S", gmtime())
# try:
#     os.stat('./cp/{}'.format(times))
# except:
#     os.mkdir('./cp/{}'.format(times))

try:
    os.stat('./logs/{}'.format(times))
except:
    os.mkdir('./logs/{}'.format(times))

checkpoint = ModelCheckpoint(filepath='./cp/lions-a02-{epoch:02d}-{val_loss:.2f}.hdf5')

tensorboard = TensorBoard(log_dir='./logs/{}/'.format(times),
                          histogram_freq=2,
                          write_graph=False,
                          write_images=True)

train_generator = kir_generator(img_dir="../Sealion/Train", coord_file= "../Sealion/coords.csv", b_size=batch_size)
valid_generator = kir_generator(img_dir="../Sealion/Train", coord_file= "../Sealion/coords.csv", b_size=batch_size)

# ========= Model ==========

# Test generator
# gen = kir_generator(img_dir="../Sealion/Train", coord_file="../Sealion/coords.csv", b_size=1)
# for j in range(15):
#     a, b = next(gen)
#     plt.subplot(231)
#     plt.imshow((a[0, :, :, :].squeeze()+1)/2);
#     plt.subplot(232)
#     plt.imshow(b[0, :, :, 0].squeeze());
#     plt.subplot(233)
#     plt.imshow(b[0, :, :, 1].squeeze());
#     plt.subplot(234)
#     plt.imshow(b[0, :, :, 2].squeeze());
#     plt.subplot(235)
#     plt.imshow(b[0, :, :, 3].squeeze());
#     plt.subplot(236)
#     plt.imshow(b[0, :, :, 4].squeeze());
#     plt.show()

if resumeFrom == None:


    model = UNet().create_model(img_shape=(img_width, img_height, 3), use_model='cp/crops-65-0.53.hdf5', pop_layers=2)
    # model = UNet().create_model(img_shape=(img_width, img_height, 3), use_model='VGG16')
    print_summary(model)

    # Freeze the first few layers which we don't want to train
    for layer in model.layers[:16]:
        layer.trainable = False


    # compile the model
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-08, decay=0.0005), metrics=['accuracy'])
    # model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"], decay=0.0005)

    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizers.RMSprop(lr=0.00003, rho=0.9, epsilon=1e-08, decay=0.000001),
    #               metrics=['accuracy'])

    # model.compile(loss = "categorical_crossentropy",
    #               optimizer = optimizers.SGD(lr=0.00003, momentum=0.9, nesterov=True, decay=1e-6),
    #               metrics=["accuracy"],
    #               decay=0.0005)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Nadam(lr=0.000005),
                  metrics=['accuracy'])

    # model.compile(loss = "mean_absolute_error",
    #               optimizer = optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True),
    #               metrics=["accuracy"],
    #               decay=0.0005)

    print_summary(model)
    plot_model(model, to_file='model.pdf', show_shapes=True)

    history = model.fit_generator(
            train_generator,
            steps_per_epoch=200,
            epochs=epochs1,
            validation_data=valid_generator,
            validation_steps=50,
            initial_epoch=0,
            pickle_safe=True,
            verbose=1,
            callbacks=[checkpoint, tensorboard])

    kir_save_history(history, 'pre')
