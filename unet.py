from keras.models import Model, load_model
from keras.layers import UpSampling2D, Cropping2D, ZeroPadding2D, concatenate
from keras.applications import VGG16
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Conv2D, Reshape, Input, Dense
from keras.utils.layer_utils import print_summary

class UNet:
    def __init__(self):
        print('Build U-Net ...')

    def get_crop_shape(self, target, refer):

        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw / 2), int(cw / 2) + 1
        else:
            cw1, cw2 = int(cw / 2), int(cw / 2)

        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch / 2), int(ch / 2) + 1
        else:
            ch1, ch2 = int(ch / 2), int(ch / 2)

        return (ch1, ch2), (cw1, cw2)

    def create_model(self, img_shape=None, use_model='', pop_layers=0):

        # use_model could be either 'VGG16' or 'RESNET' or path to a checkpoint. Set pop_layers to the number of layers to strip

        # inputs = Input(shape=img_shape)
        # concat_axis = 1

        # load base VGG model
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=img_shape)
        print(' VGG Model loaded.')

        # make the structure fully match to the crop-trained net
        x = base_model.output
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Conv2D(512, (1, 1), activation='elu', padding='valid', name='Kir_1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (1, 1), activation='elu', padding='valid', name='Kir_2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(6, (3, 3), activation='sigmoid', padding='valid', name='Kir_3')(x)
        # predictions = Reshape(target_shape=(6,))(x)

        base_model = Model(input=base_model.input, output=x)        # это уже совсем другая base_model

        # now load weights
        base_model.load_weights(use_model)

        # remove un-neded layers
        for i in range(pop_layers):
            base_model.layers.pop()

        print(' Custom Model loaded and trimmed.')
        # base_model.summary()

        # Freeze the first few layers which we don't want to train
        # for layer in modelVGG.layers[:16]:
        #     layer.trainable = False

        concat_axis = 3

        base_layer_outputs = {}

        for layer in base_model.layers:
            layer.trainable = False
            base_layer_outputs[layer.name] = layer.get_output_at(0)
            # print(layer.name)

        x = base_model.output
        print_summary(base_model)

        # now upsampling
        up_conv1 = UpSampling2D(size=(2, 2), name='up_conv1')(x)

        ch, cw = self.get_crop_shape(base_layer_outputs['block5_conv3'], up_conv1)
        fw_layer = Cropping2D(cropping=(ch, cw))(base_layer_outputs['block5_conv3'])
        up_conv = concatenate([up_conv1, fw_layer], axis=concat_axis)

        up_conv = Dropout(0.2)(up_conv)

        up_conv = Conv2D(512, (3, 3), activation='elu', padding='same')(up_conv)
        up_conv = BatchNormalization()(up_conv)
        up_conv = Conv2D(512, (3, 3), activation='elu', padding='same')(up_conv)
        up_conv = BatchNormalization()(up_conv)

        up_conv2 = UpSampling2D(size=(2, 2))(up_conv)
        ch, cw = self.get_crop_shape(base_layer_outputs['block4_conv3'], up_conv2)
        fw_layer = Cropping2D(cropping=(ch, cw))(base_layer_outputs['block4_conv3'])
        up_conv = concatenate([up_conv2, fw_layer], axis=concat_axis)

        up_conv = Dropout(0.2)(up_conv)
        up_conv = Conv2D(256, (3, 3), activation='elu', padding='same')(up_conv)
        up_conv = BatchNormalization()(up_conv)
        up_conv = Conv2D(256, (3, 3), activation='elu', padding='same')(up_conv)
        up_conv = BatchNormalization()(up_conv)

        up_conv3 = UpSampling2D(size=(2, 2))(up_conv)
        ch, cw = self.get_crop_shape(base_layer_outputs['block3_conv3'], up_conv3)
        fw_layer = Cropping2D(cropping=(ch, cw))(base_layer_outputs['block3_conv3'])
        up_conv = concatenate([up_conv3, fw_layer], axis=concat_axis)

        up_conv = Dropout(0.2)(up_conv)
        up_conv = Conv2D(128, (3, 3), activation='elu', padding='same')(up_conv)
        up_conv = BatchNormalization()(up_conv)
        up_conv = Conv2D(128, (3, 3), activation='elu', padding='same')(up_conv)

        # output X x X
#        output = Conv2D(5, (1, 1), activation='sigmoid', padding='same')(up_conv)
        output = Dense(5, activation='softmax')(up_conv)

        model = Model(input=base_model.input, output=output)
        model.summary()

        return model
