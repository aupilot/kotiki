import math
import pickle
import time
from contextlib import contextmanager

import skimage.io
import skimage.transform
import zstd
from skimage.transform import SimilarityTransform, AffineTransform
import numpy as np


def crop_edge(img, x, y, w, h, mode='edge'):
    img_w = img.shape[1]
    img_h = img.shape[0]

    if x >= 0 and y >= 0 and x + w <= img_w and y + h < img_h:
        return img[int(y):int(y + h), int(x):int(x + w)].astype('float32') / 255.0

    tform = SimilarityTransform(translation=(x, y))
    return skimage.transform.warp(img, tform, mode=mode, output_shape=(h, w))


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l) // n * n + n - 1, n):
        if len(l[i:i + n]):
            yield l[i:i + n]


def load_compressed_data(file_name):
    with open(file_name, 'rb') as f:
        compressed = f.read()


    data = pickle.loads(zstd.decompress(compressed))
    # data = compressed

    # comp_ctx = zstd.ZstdDecompressor()
    # data = pickle.loads(comp_ctx.decompress(compressed))
    # del comp_ctx
    return data


def save_compressed_data(data, file_name):
    # comp_ctx = zstd.ZstdCompressor(write_content_size=True, level=6)
    # compressed = comp_ctx.compress(pickle.dumps(data, protocol=-1))

    compressed = zstd.compress(pickle.dumps(data, protocol=-1))
    # compressed = pickle.dumps(data, protocol=-1)

    f = open(file_name, 'wb')
    f.write(compressed)
    f.close()


def lock_layers_until(model, first_trainable_layer, verbose=False):
    found_first_layer = False
    for layer in model.layers:
        if layer.name == first_trainable_layer:
            found_first_layer = True

        if verbose and found_first_layer and not layer.trainable:
            print('Make layer trainable:', layer.name)
            layer.trainable = True

        layer.trainable = found_first_layer


def get_image_crop(full_rgb, rect, scale_rect_x=1.0, scale_rect_y=1.0,
                   shift_x_ratio=0.0, shift_y_ratio=0.0,
                   angle=0.0, out_size=299):
    center_x = rect.x + rect.w / 2
    center_y = rect.y + rect.h / 2
    size = int(max(rect.w, rect.h))
    size_x = size * scale_rect_x
    size_y = size * scale_rect_y

    center_x += size * shift_x_ratio
    center_y += size * shift_y_ratio

    scale_x = out_size / size_x
    scale_y = out_size / size_y

    out_center = out_size / 2

    tform = AffineTransform(translation=(center_x, center_y))
    tform = AffineTransform(rotation=angle * math.pi / 180) + tform
    tform = AffineTransform(scale=(1 / scale_x, 1 / scale_y)) + tform
    tform = AffineTransform(translation=(-out_center, -out_center)) + tform
    return skimage.transform.warp(full_rgb, tform, mode='edge', order=1, output_shape=(out_size, out_size))


def crop_zero_pad(img, x, y, w, h):
    img_w = img.shape[1]
    img_h = img.shape[0]

    if x >= 0 and y >= 0 and x + w <= img_w and y + h < img_h:
        return img[int(y):int(y + h), int(x):int(x + w)]
    else:
        res = np.zeros((h, w)+img.shape[2:])
        x_min = int(max(x, 0))
        y_min = int(max(y, 0))
        x_max = int(min(x + w, img_w))
        y_max = int(min(y + h, img_h))
        res[y_min - y:y_max-y, x_min - x:x_max-x] = img[y_min:y_max, x_min:x_max]
        return res


def generate_overlapped_crops(img, crop_w, crop_h, overlap):
    img_h, img_w = img.shape[:2]
    n_h = int(np.ceil((img_h + overlap) / (crop_h - overlap)))
    n_w = int(np.ceil((img_w + overlap) / (crop_w - overlap)))

    res = np.zeros((n_w*n_h, crop_h, crop_w, ) + img.shape[2:])

    for i_h in range(n_h):
        for i_w in range(n_w):
            x = -overlap + i_w * (crop_w - overlap)
            y = -overlap + i_h * (crop_h - overlap)
            res[i_h * n_w + i_w] = crop_zero_pad(img, x, y, crop_w, crop_h)

    return res


def print_stats(title, array):
    print('{} shape:{} dtype:{} min:{} max:{} mean:{} median:{}'.format(
        title,
        array.shape,
        array.dtype,
        np.min(array),
        np.max(array),
        np.mean(array),
        np.median(array)
    ))

if __name__ == '__main__':
    pass
    #test_chunks()
    #
    # img = skimage.io.imread('../train/ALB/img_00003.jpg')
    # print(img.shape)
    #
    # with timeit_context('Generate crops'):
    #     crop_edge(img, 10, 10, 400, 400)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure(1)
    # plt.subplot(221)
    # plt.imshow(img)
    # plt.subplot(222)
    # plt.imshow(crop_edge(img, 1280-200, 720-200, 400, 400, mode='edge'))
    # plt.subplot(223)
    # plt.imshow(crop_edge(img, 1280 - 200, 720 - 200, 400, 400, mode='wrap'))
    # plt.subplot(224)
    # plt.imshow(crop_edge(img, 1280 - 200, 720 - 200, 400, 400, mode='reflect'))
    #
    # plt.show()
