import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from PIL import Image
import argparse
import rasterio

parser = argparse.ArgumentParser(description='Script for running super resolution')
parser.add_argument('--image_path', required=True,
    help='Dataset to use', type=str)
parser.add_argument('--algorithm', default='edsr')
parser.add_argument('--resolution_factor', required=True, type=int)
parser.add_argument('--out_file', required=True, type=str)

args = parser.parse_args()

"""
NOTE: this script is optimized for TF1. Use the bva-tf.sif environment
"""
def main():
    log = open("log.txt", "a")
    # from tensorflow.python.client import device_lib
    # def get_available_devices():
    #     local_device_protos = device_lib.list_local_devices()
    #     return [x.name for x in local_device_protos]
    # print(get_available_devices())
    model_path = None
    if args.resolution_factor in [2, 3, 4]:
        model_path = "../all_buildings/scripts/berkeley/checkpoints/EDSR_x{}.pb".format(args.resolution_factor)
    else:
        log.write("Incorrect zoom resolution. Please specify either 2, 3, or 4.\n")
        return


    # READ IMAGE
    image_path = args.image_path
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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    all_predictions = []
    count = 0
    final_count = len(test)

    start = time.perf_counter()
    while count < final_count:
        start_session = time.perf_counter()

        with tf.Session(config=config) as persisted_sess:
            with tf.gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                persisted_sess.graph.as_default()
                tf.import_graph_def(graph_def)
                session_start = count
                session_stop = count+10
                if session_stop > final_count:
                    session_stop = final_count

                for i in range(session_start, session_stop):
                    log.write(str(count) + '\n')

                    output = persisted_sess.graph.get_tensor_by_name('import/NCHW_output:0')
                    prediction = persisted_sess.run(output, {'import/IteratorGetNext:0': [test[i]]})
                    # prediction = np.rint(np.transpose(prediction[0], (1, 2, 0))).astype(int)

                    prediction = np.rint(prediction[0]).astype(int)

                    all_predictions.append(prediction)

                    count += 1
        session_elapsed = time.perf_counter() - start_session
        log.write('Elapsed %.3f seconds.\n' % session_elapsed)

    # stitch to one large array
    all_predictions = np.concatenate(all_predictions, axis=1)
    log.write('Number of bytes: {}\n'.format(all_predictions.nbytes))

    # with open(args.out_file, 'wb') as f:
    #     np.savez_compressed(f, all_predictions)

        # plot_sample(pic, np.rint(np.transpose(prediction[0], (1, 2, 0))).astype(int))

    # write using rasterio
    with rasterio.open(args.image_path) as src:  # open raster dataset
        out_profile = src.profile.copy()

        out_profile.update({'count': 3, 'height': all_predictions.shape[1], 'width': all_predictions.shape[2]})

    with rasterio.open(args.out_file, 'w', **out_profile) as dst:  # open raster dataset in 'w' write mode using the
        dst.write(all_predictions)

    # will probably need to change the profile

    elapsed = time.perf_counter() - start
    log.write('Elapsed %.3f seconds.\n' % elapsed)

if __name__ == "__main__":
    main()