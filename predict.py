import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pathlib

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import constants
import datasetloader
import model



if __name__ == '__main__':
    tf.random.set_seed(constants.RANDOM_SEED)
    
    DATASET_SAVED_PATH:pathlib.Path = pathlib.Path(os.path.join(pathlib.Path(__file__).parents[0], 'ModelNet10'))

    """ Load the pointcloud dataset """
    train_points, test_points, train_labels, test_labels, class_map = datasetloader.processDataset(DATASET_SAVED_PATH, constants.UNIFORM_SAMPLE_COUNT)
    print('TEST SHAPES ::: ', test_points.shape, test_labels.shape)

    print('LOAD WEIGHTS FROM A CHECKPOINT')
    checkpoint_path = pathlib.Path('./trained_model/checkpoints')

    """ Create the tensor slices of the loaded dataset (for both training and testing) """
    print('CREATEING THE TENSOR SLICES FOR THE TRAINING AND TESTING DATASET')
    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

    """ Apply shuffling and agumentation to the training dataset """
    print('APPLYING THE AUGMENTATION FOR THE TRAINING AND TESTING DATASETS')
    train_dataset = train_dataset.shuffle(len(train_points)).map(datasetloader.augmentation).batch(constants.BATCH_SIZE)
    """ Apply shuffling only for the testing dataset """
    test_dataset = test_dataset.shuffle(len(test_points)).batch(constants.BATCH_SIZE)

    """ Create the pointnet model """
    print('INITIALIZE THE POINTNET MODEL')
    model = model.createPointNetModel(constants.UNIFORM_SAMPLE_COUNT, constants.NUM_CLASSES)

    latest = tf.train.latest_checkpoint(checkpoint_path)
    # Load the previously saved weights
    model.load_weights(latest)

    data = test_dataset.take(1)
    points, labels = list(data)[0]
    print('LABELS ::: ', test_labels.shape)
    print('POINTS ::: ', points.shape)
    print('LABELS :::', labels.shape)
    points = points[:8, ...]
    labels = labels[:8, ...]

    # raise IndentationError

    # run test data through model
    preds = model.predict(points)
    preds = tf.math.argmax(preds, -1)

    points = points.numpy()

    # plot points with predicted class and label
    fig = plt.figure(figsize=(15, 10))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        print('TOTAL POINTS ::: ', points[i].shape, preds[i].numpy())
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
        ax.set_title(
            "pred: {:}, label: {:}".format(
                class_map[preds[i].numpy()], class_map[labels.numpy()[i]]
            )
        )
        ax.set_axis_off()
    plt.show()