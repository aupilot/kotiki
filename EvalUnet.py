import numpy as np
import os
import re
from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


pad_width, pad_height   = 50, 50              # size for padding of tales надо понять насколько они должны быть большие!
tale_width, tale_height = 500, 500            # size of tales before padding
img_width, img_height   = tale_width + pad_width, tale_height + pad_height


model = load_model("./cp/lions-a01-14-0.00.hdf5")


def eval_generator(img_dir):
    global pad_width, pad_height
    global tale_width, tale_height
    global img_width, img_height
    global out_width, out_height

    half_w = int(pad_width  / 2)
    half_h = int(pad_height / 2)

    file_names = [f for f in os.listdir(img_dir) if re.match(r'[0-9]+.*\.jpg', f)]
    file_names = sorted(file_names, key=lambda
        item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))

    X = np.zeros((img_width, img_height, 3))

    ptr = 0

    while True:
        for filename in file_names:
            img = load_img(img_dir+'/'+filename)
            width, height = img.size
            img_resized = img.resize((4500, 3000))
            k = 3000./float(height)

            # special type image to np array
            x = img_to_array(img_resized)
            x = x/128.-1.

            # tale the picture to 6 pcs
            for cy in range(0, int(3000/tale_height)):
                for cx in range(0, int(4500/tale_width)):
                    # make a padded image from crop
                    xx = x[cy * tale_height:(cy + 1) * tale_height, cx * tale_width:(cx + 1) * tale_width, :]
                    # now we add borders around the tale
                    X  = np.pad(xx, ((half_h,half_h), (half_w, half_w), (0,0)), mode='edge')
                    XX = np.expand_dims(X, axis=0)     # we need one more dimension for batch size

                    yield XX

gen = eval_generator(img_dir="../Sealion/Train")
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

for i in range(100):
    img = next(gen)
    prediction = model.predict(img, batch_size=1)
    aaa = prediction.squeeze()
    plt.subplot(2, 1, 1)
    plt.imshow((img.squeeze()+1.)/2.)
    plt.subplot(2, 1, 2)
    plt.imshow(aaa[:, :, 3])
    plt.show()
