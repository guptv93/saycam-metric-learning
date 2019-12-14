'''Moving MNIST generation code (modified from code by Prateek Mahajan:
https://gist.github.com/praateekmahajan/b42ef0d295f528c986e2b3a0b31ec1fe)'''
import math
import os
import sys
import numpy as np
import mnist
import matplotlib.pylab as plt
from PIL import Image


COLOR_MAT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])


def get_image_from_array(X, index, mean=0, std=1):
    '''
    Args:
        X: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    '''
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = (((X[index] + mean) * 255.) * std).reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret


def generate_moving_mnist(training, shape=(64, 64), num_frames=30, num_images=100, original_size=28, nums_per_image=2,
                          color=True, noise=0.0):
    '''
    Args:
        training: Boolean, used to decide if downloading/generating train set or test set
        shape: Shape we want for our moving images (new_width and new_height)
        num_frames: Number of frames in a particular movement/animation/gif
        num_images: Number of movement/animations/gif to generate
        original_size: Real size of the images (eg: MNIST is 28 x 28)
        nums_per_image: Digits per animation.
    Returns:
        Dataset of np.uint8 type with dimensions num_frames * num_images x 1 x new_width x new_height
    '''
    global COLOR_MAT

    x_train, t_train, x_test, t_test = mnist.load()
    x_train = x_train / 255
    x_test = x_test / 255

    if training:
        X = x_train
        Y = t_train
    else:
        X = x_test
        Y = t_test

    X = np.reshape(X, (-1, original_size, original_size))

    width, height = shape

    # Get how many pixels can we move around a single image
    lims = (x_lim, y_lim) = width - original_size, height - original_size

    # Create a dataset
    frames = np.float32(noise * np.random.rand(num_frames * num_images, width, height, 3))
    labels = np.empty((num_frames * num_images, nums_per_image), dtype=np.uint8)
    colors = np.empty((num_frames * num_images, nums_per_image), dtype=np.uint8)

    for img_idx in range(num_images):
        # Randomly generate direction, speed and velocity for both images
        direcs = np.pi * (np.random.rand(nums_per_image) * 2 - 1)
        speeds = np.random.randint(5, size=nums_per_image) + 2
        veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])

        indices = np.random.randint(0, X.shape[0], nums_per_image)
        curr_X = X[indices]
        curr_Y = Y[indices]

        labels[(img_idx*num_frames):((img_idx+1)*num_frames), :] = curr_Y

        new_X = np.empty((nums_per_image, original_size, original_size, 3), dtype=np.float32)
        if color:
            # colors
            curr_color = np.random.randint(0, COLOR_MAT.shape[0], nums_per_image)
        else:
            curr_color = 3 * np.ones(nums_per_image, dtype=int)

        for i in range(nums_per_image):
            new_X[i, :, :, 0] = curr_X[i, :, :] * COLOR_MAT[curr_color[i], 0]
            new_X[i, :, :, 1] = curr_X[i, :, :] * COLOR_MAT[curr_color[i], 1]
            new_X[i, :, :, 2] = curr_X[i, :, :] * COLOR_MAT[curr_color[i], 2]

        colors[(img_idx*num_frames):((img_idx+1)*num_frames), :] = curr_color

        # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
        positions = np.asarray([(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(nums_per_image)])

        # Generate new frames for the entire num_frames
        for frame_idx in range(num_frames):

            positions = positions.astype(int)

            for i in range(nums_per_image):
                # Add the canvas to the dataset array, ceheck this part
                start_x, end_x = positions[i, 0], positions[i, 0] + original_size
                start_y, end_y = positions[i, 1], positions[i, 1] + original_size

                frames[img_idx * num_frames + frame_idx, start_x:end_x, start_y:end_y, :] += new_X[i]

            # Get the next position by adding velocity
            next_pos = positions + veloc

            # Iterate over velocity and see if we hit the wall
            # If we do then change direction
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -1 or coord > lims[j] + 1:
                        veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))

            # Make the permanent change to position by adding updated velocity
            positions = positions + veloc


    return frames, labels, colors


def main(training, dest, filetype='npz', frame_size=64, num_frames=30, num_images=100, original_size=28,
         nums_per_image=2, color=True, noise=0.0, plot=False):
    frames, labels, colors = generate_moving_mnist(training, shape=(frame_size, frame_size), num_frames=num_frames,
                                                    num_images=num_images, original_size=original_size,
                                                    nums_per_image=nums_per_image, color=color, noise=noise)

    frames = frames.clip(0, 1).astype(np.float32)
    
    if filetype == 'npy':
        dest = "mm_folder2"
        for i in range(num_images):
            os.mkdir(dest+"/%s" %i)
            np.save(dest+"/%s/np_%s" % (i,i), frames[(i)*num_frames : (i+1)*num_frames])
    elif filetype == 'npz':
        np.savez(dest, frames=frames, labels=labels, colors=colors)
    elif filetype == 'jpg':
        for i in range(frames.shape[0]):
            Image.fromarray(get_image_from_array(frames, i, mean=0)).save(os.path.join(dest, '{}.jpg'.format(i)))

    if plot:
        plot_movingmnist(frames)

# fix this
def plot_movingmnist(frames):

    for i in range(frames.shape[0]):
        plt.imshow(frames[i, :, :, :], interpolation='nearest')
        plt.pause(.01)
        plt.clf()

    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--dest', type=str, dest='dest', default='movingmnistdata')
    parser.add_argument('--filetype', type=str, dest='filetype', default="npy")
    parser.add_argument('--training', type=bool, dest='training', default=True)
    parser.add_argument('--frame_size', type=int, dest='frame_size', default=224)
    parser.add_argument('--num_frames', type=int, dest='num_frames', default=30)  # length of each sequence
    parser.add_argument('--num_images', type=int, dest='num_images', default=2)  # number of sequences to generate
    parser.add_argument('--original_size', type=int, dest='original_size', default=28)  # size of mnist digit within frame
    parser.add_argument('--nums_per_image', type=int, dest='nums_per_image', default=1)  # number of digits in each frame
    parser.add_argument('--color', type=bool, dest='color', default=True)  # whether to use digits with color
    parser.add_argument('--noise', type=float, dest='noise', default=0.0)  # noise level

    args = parser.parse_args(sys.argv[1:])
    main(**{k: v for (k, v) in vars(args).items() if v is not None})