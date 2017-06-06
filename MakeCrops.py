# create a set of crops for further learning

import os
import cv2
import numpy as np
import re, math, random, time
from scipy.spatial.distance import cdist


def makeSquare(thumb, lionSize):
    """
    pad an image to make it square

    """
    h, w, c = np.shape(thumb)
    cv2.copyMakeBorder(thumb,
                       math.floor((lionSize - h) / 2),
                       math.ceil((lionSize - h) / 2),
                       math.floor((lionSize - w) / 2),
                       math.ceil((lionSize - w) / 2),
                       cv2.BORDER_REFLECT_101)
    return thumb

lion_size = 112

out_dir_classes = True               #  use classification dir structure (for built-in classification generatior)
out_dir = "./train."+str(lion_size)+"/" if out_dir_classes else "./train.scale."+str(lion_size)+"/"

inp_dir = '../Sealion/'
classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups", "negatives"]
coords = "/Volumes/KProData/Users/kir/Documents/Programming/Python/Sealion/coords.csv"

start_time = time.time()

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# load CSV
centres = np.genfromtxt(coords, delimiter=',', skip_header=1, dtype=int)
# centres = pd.read_csv(coords)

file_names = os.listdir(inp_dir + "Train/")
file_names = sorted(file_names, key=lambda
    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))

ds_store = ".DS_Store"
if ds_store in file_names: file_names.remove(ds_store)


# select a subset of files to run on
# file_names = file_names[1:-1]

# dataframe to store results in
# coordinates_df = pd.DataFrame(index=file_names, columns=classes)

for filename in file_names:

    if 'jpg' not in filename:
        continue

    # read a Train image
    image = cv2.imread(inp_dir + "Train/" + filename)
    height, width, channels = np.shape(image)

    print(filename + '\t', end="", flush=True)

    fileNoTxt = re.findall(r'\d+', filename)
    fileNo = int(fileNoTxt[0])

    idx = (centres[:, 0] == fileNo)
    centres1img = centres[idx]

    pos = 0         # считаем positive samples
    neg = 0         # считаем negative samples
    num = 0         # сквозная нумерация для случая без out_dir_classes - включаем в crop file name

    for lion_class in classes:
        iii = 0
        if lion_class != 'negatives':
            # === positives
            idx = (centres1img[:, 1] == classes.index(lion_class))
            centresClass = centres1img[idx]

            for cent in centresClass:
                # in coords y -> [2], x -> [3]
                thumb = image[cent[2] - int(lion_size / 2):cent[2] + int(lion_size / 2),
                        cent[3] - int(lion_size / 2):cent[3] + int(lion_size / 2), :]

                # if not square - discard
                if np.shape(thumb) != (lion_size, lion_size, 3):
                    continue

                # print(np.shape(thumb))
                # plt.imshow(thumb);plt.draw()
                # plt.show(block=False)

                if out_dir_classes:
                    cv2.imwrite(out_dir + lion_class + '/' + str(fileNo) + "_" + str(iii) + ".jpg", thumb)
                else:
                    cv2.imwrite(out_dir + str(fileNo) + "_" + str(num) + ".jpg", thumb)

                pos = pos + 1
                iii = iii + 1
                num = num + 1

        else:
            # === negatives
            # make random crops ~ twice as much as another classes, given that they are far away from any centres
            i = 0
            while True:
                # generate neg centre
                randCent = np.array([(random.randint(int(lion_size / 2), width - int(lion_size / 2)),
                                      random.randint(int(lion_size / 2), height - int(lion_size / 2)))])

                # find the distance to all positive centres in the picture
                allCents = np.array(centres1img[:, [2, 3]])
                dist = cdist(allCents, randCent, 'sqeuclidean')

                # if the picrture has no lions
                if dist.size == 0:
                    # just save few random
                    thumb = image[randCent[0, 1] - int(lion_size / 2):randCent[0, 1] + int(lion_size / 2),
                            randCent[0, 0] - int(lion_size / 2):randCent[0, 0] + int(lion_size / 2), :]

                    if out_dir_classes:
                        # make it square if not
                        if np.shape(thumb) != (lion_size, lion_size, 3):
                            thumb = makeSquare(thumb, lion_size)
                        cv2.imwrite(out_dir + lion_class + '/' + str(fileNo) + "_" + str(iii) + ".jpg", thumb)
                    else:
                        # if not square - discard
                        if np.shape(thumb) != (lion_size, lion_size, 3):
                            continue
                        cv2.imwrite(out_dir + str(fileNo) + "_" + str(num) + ".jpg", thumb)

                    neg = neg + 1
                    iii = iii + 1
                    num = num + 1
                else:
                    # save only if not too close & not too far from positive lioms (or randomly with prob 5%)
                    if ((np.amin(dist) > lion_size / 3) and (np.amax(dist) < (lion_size * 2) ** 2)) or random.randint(1,
                                                                                                                      100) < 5:
                        # print(dist)
                        thumb = image[randCent[0, 1] - int(lion_size / 2):randCent[0, 1] + int(lion_size / 2),
                                randCent[0, 0] - int(lion_size / 2):randCent[0, 0] + int(lion_size / 2), :]

                        if out_dir_classes:
                            # make it square if not
                            if np.shape(thumb) != (lion_size, lion_size, 3):
                                thumb = makeSquare(thumb, lion_size)
                            cv2.imwrite(out_dir + lion_class + '/' + str(fileNo) + "_" + str(iii) + ".jpg", thumb)
                        else:
                            # if not square - discard
                            if np.shape(thumb) != (lion_size, lion_size, 3):
                                continue
                            cv2.imwrite(out_dir + str(fileNo) + "_" + str(num) + ".jpg", thumb)

                        neg = neg + 1
                        iii = iii + 1
                        num = num + 1

                # don't want to stuck here forever
                i = i + 1
                if i > 2500:
                    break
                if pos != 0 and neg >= int(pos / 3):
                    break
                if pos == 0 and neg >= 10:
                    break

        print(".", end="", flush=True)
    print(" Pos:", pos, " Neg:", neg)
print("--- %s seconds ---" % (time.time() - start_time))





