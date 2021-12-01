import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from PIL import Image

print(tf.__version__)
from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
# print(get_available_devices())

model_path = "../all_buildings/scripts/berkeley/checkpoints/EDSR_x4.pb"


# READ IMAGE
image_path = "/oak/stanford/groups/deho/building_compliance/berkeley_naip_2020/berkeley_ne.tif"
img = np.array(Image.open(image_path))
img = img[:, :, :3]
test = np.array_split(img, 100, axis=1)
test = [np.array_split(i, 100, axis=0) for i in test]

def plot_sample(lr, sr):
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', 'SR']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img[:, :, ::-1])
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

    plt.savefig('test.png')


pic = test[50][80]
# tf.debugging.set_log_device_placement(True)

start = time.perf_counter()
with tf.Session() as persisted_sess:
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def)

        output = persisted_sess.graph.get_tensor_by_name('import/NCHW_output:0')
        prediction = persisted_sess.run(output, {'import/IteratorGetNext:0': [pic]})

    plot_sample(pic, np.rint(np.transpose(prediction[0], (1, 2, 0))).astype(int))

elapsed = time.perf_counter() - start
print('Elapsed %.3f seconds.' % elapsed)