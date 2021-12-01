import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from PIL import Image
import argparse
# apparently args is not a package that is installed?

# parser = argparse.ArgumentParser(description='Script for running super resolution')
# parser.add_argument('--image_path', required=True,
#     help='Dataset to use'
# )
# parser.add_argument('--algorithm', default='kl',
#     choices=(
#         'kl',
#         'color'
#     ),
#     help='Algorithm to use'
# )
#
# parser.add_argument('--num_clusters', type=int, required=False, help='Number of clusters to use in k-means step.')
#
# parser.add_argument('--parcel_type', required=True, default='no_parcel',
#                     choices = ('no_parcel', 'parcel', 'parcel_dedup'),
#                     help='Specify if using parcel type, dedup, etc.',)
#
# group = parser.add_mutually_exclusive_group(required=True)
# group.add_argument('--buffer', type=float, help='Amount to buffer for defining a neighborhood. Note: this will be in terms of units of the dataset.')
#
# parser.add_argument('--output_dir', type=str, required=True, help='Path to an empty directory where outputs will be saved. This directory will be created if it does not exist.')
# parser.add_argument('--verbose', action="store_true", default=False, help='Enable training with feature disentanglement')
# parser.add_argument('--overwrite', action='store_true', default=False, help='Ignore checking whether the output directory has existing data')
#
# args = parser.parse_args()

"""
NOTE: this script is optimized for TF1. Use the bva-tf.sif environment
"""

print(tf.__version__)
# from tensorflow.python.client import device_lib
# def get_available_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos]
# print(get_available_devices())

model_path = "../all_buildings/scripts/berkeley/checkpoints/EDSR_x4.pb"


# READ IMAGE
image_path = "/oak/stanford/groups/deho/building_compliance/berkeley_naip_2020/berkeley_ne.tif"
img = np.array(Image.open(image_path))
img = img[:, :, :3] # need to subset for just RBG bands
test = np.array_split(img, 100, axis=0)
# test = [np.array_split(i, 10, axis=0) for i in test]

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

    # plt.savefig('test1.png')


# pic = img
# print(pic.shape)

start = time.perf_counter()
with tf.Session(config=config) as persisted_sess:
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def)

        all_predictions = []
        count = 1
        for i in test:
            print(i.shape)
            # tbh, not sure if i can for loop like this within the persisted_sess--maybe
            output = persisted_sess.graph.get_tensor_by_name('import/NCHW_output:0')
            prediction = persisted_sess.run(output, {'import/IteratorGetNext:0': [i]})
            prediction = np.rint(np.transpose(prediction[0], (1, 2, 0))).astype(int)
            all_predictions.append(prediction)
            print(count)
            count += 1

        # stitch to one large array
        all_predictions = np.concatenate(all_predictions, axis=0)

        with open('berkeley_ne.npz', 'wb') as f:
            np.savez(f, all_predictions)

    # plot_sample(pic, np.rint(np.transpose(prediction[0], (1, 2, 0))).astype(int))

elapsed = time.perf_counter() - start
print('Elapsed %.3f seconds.' % elapsed)