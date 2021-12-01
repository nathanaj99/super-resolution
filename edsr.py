import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

print(tf.__version__)
print(tf.config.list_physical_devices())

model_path = "/all_buildings/scripts/berkeley/checkpoints/EDSR_x4.pb"


# READ IMAGE
img = cv2.imread("/oak/stanford/groups/deho/building_compliance/berkeley_naip_2020/berkeley_ne.tif")

test = np.array_split(img, 100, axis=1)
test = [np.array_split(i, 100, axis=0) for i in test]

def plot_sample(lr, sr):
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img[:, :, ::-1])
        plt.title(title)
        plt.xticks([])
        plt.yticks([])


pic = test[50][80]
tf.debugging.set_log_device_placement(True)

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