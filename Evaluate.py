import os
import matplotlib.pyplot as plt
import cv2, numpy as np

from keras.utils.layer_utils import print_summary
from keras.preprocessing.image import load_img, img_to_array
from keras import applications

from keras.optimizers import SGD

from keras.models import load_model

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D


classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups", "negatives"]

test_image = "../Sealion/Train/5.jpg"
#test_image = "validation/adult_females/8_44.jpg"

#img_width, img_height = 2000, 2000
img_width, img_height = 1120, 1120
#img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)


def create_model(weights_path=None, inp_shape=None):

    # img_input = Input(shape=inp_shape)

    # model = applications.VGG16(weights=None, include_top=False, input_shape=inp_shape)
    model = applications.ResNet50(weights=None, include_top=False, input_shape=inp_shape)

    pretr_layer_outputs = {}
    for layer in model.layers:
        # layer.trainable = False
        pretr_layer_outputs[layer.name] = layer.get_output_at(0)
    x = pretr_layer_outputs['activation_49']  # we need to skip the very last layer in case of ResNet if 224x224

    # here are my layers
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
    x = Conv2D(6, (1, 1), activation='softmax', padding='valid', name='Kir_30')(x)   # возможно это полезно довести 3х3 до конца, чтобы задавить соседних котиков
    # predictions = Reshape(target_shape=(6,))(x)


    # Create model.
    # model = Model(img_input, x, name='kir_a')
    model = Model(model.input, x, name='kir_b')

    #print_summary(model)

    # load weights
    model.load_weights(weights_path, by_name=True)

    return model


# load traned net
# model = create_model("./cp/12-crops-197-0.37.hdf5", input_shape)
model = create_model("./cp/12-crops-28-0.75.hdf5", input_shape)


print_summary(model)

# load a test sample
img = load_img(test_image)
x   = img_to_array(img)             # special type image to np array
x   = x[0:img_height, 2800:2800+img_width]
y   = np.expand_dims(x, axis=0)     # we need one more dimension for batch size

prediction = model.predict(y, batch_size=1)

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
# print(classes)
# print(prediction)

# plt.figure(1);
plt.subplot(3, 3, 1); plt.imshow(x); #plt.show(block=False)
aa = prediction.squeeze()
plt.subplot(3, 3, 2); plt.imshow(aa[:,:,0]); plt.title('adult_males'); #plt.show(block=False)
plt.subplot(3, 3, 3); plt.imshow(aa[:,:,1]); plt.title('subadult_males');# plt.show(block=False)
plt.subplot(3, 3, 4); plt.imshow(aa[:,:,2]); plt.title('adult_females'); #plt.show(block=False)
plt.subplot(3, 3, 5); plt.imshow(aa[:,:,3]); plt.title('juveniles'); #plt.show()
plt.subplot(3, 3, 6); plt.imshow(aa[:,:,4]); plt.title('pups')
plt.subplot(3, 3, 7); plt.imshow(aa[:,:,5]); plt.title('Negatives')
plt.subplot(3, 3, 8); plt.imshow(aa[:,:,3]-aa[:,:,5]); plt.title('Juv-Neg'); plt.show()