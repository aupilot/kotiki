import numpy as npimport osimport refrom keras.models import Model, load_modelfrom keras.preprocessing.image import load_img, img_to_arrayfrom keras import applicationsfrom keras.layers import Dropout, Conv2D, Reshape, Dense, Flattenfrom keras.layers.normalization import BatchNormalizationfrom keras.applications.resnet50 import preprocess_inputimport matplotlib.pyplot as pltimg_width, img_height   = 224*5, 224*5# generates a crop from one imagedef eval_generator(img_dir):    file_names = [f for f in os.listdir(img_dir) if re.match(r'[0-9]+.*\.jpg', f)]    file_names = sorted(file_names, key=lambda        item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))    for filename in file_names:        img = load_img(os.path.join(img_dir, filename))        width, height = img.size        x, y = np.random.randint(width-img_width), np.random.randint(height-img_height)        # special type image to np array        dd = img_to_array(img)        ss = dd[y:y+img_height, x:x+img_width]        ss = preprocess_input(ss)        yield ss    yield np.array([[]])def create_model(weights_path=None, inp_shape=None):    # img_input = Input(shape=inp_shape)    model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=inp_shape)    pretr_layer_outputs = {}    for layer in model.layers:        pretr_layer_outputs[layer.name] = layer.get_output_at(0)    if model.name == 'resnet50':        x = pretr_layer_outputs['activation_49']       # we need to skip the very last layer in case of ResNet    else:        x = model.output    # model_pretrained.summary()    x = Conv2D(512, (7, 7), activation='elu', padding='valid', name='Kir_0')(x)    # x = AvgPool2D(pool_size=(7, 7), padding='same', name='Kir_0')(x)    # x = MaxPool2D(pool_size=(7, 7), padding='same', name='Kir_0')(x)    x = Dropout(0.25)(x)    x = BatchNormalization()(x)    x = Conv2D(256, (1, 1), activation='elu', padding='valid', name='Kir_2')(x)    x = BatchNormalization()(x)    x = Conv2D(1, (1, 1), activation='elu', padding='valid', name='Kir_3')(x)    # x = Reshape(target_shape=(1,))(x)   # Create model.    # model = Model(img_input, x, name='kir_a')    model = Model(model.input, x, name='kir_b')    # print_summary(model)    # load weights    model.load_weights(weights_path)    return modelmodel = create_model(weights_path="./cp/scale-06-0.17.hdf5", inp_shape=(img_width, img_height, 3) )gen = eval_generator(img_dir="../Sealion/Train")np.set_printoptions(formatter={'float': '{: 0.4f}'.format})# while True:for i in range(10):    img = next(gen)    if img.size == 0:        break    yy = np.expand_dims(img, axis=0)    prediction = model.predict(yy)    aaa = prediction.squeeze()    print(aaa)    plt.imshow((img.squeeze() + 1) / 2)