import matplotlib.pyplot as plt
import os

def kir_save_history(history, prefix):
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

