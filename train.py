""" IMPORTANT FOR SOME CUDA BASED MACHINES """
"""
Run the python script with the following flags before running
example,

XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python train.py
instead of just python train.py. 
Otherwise  you will have the internal libdevice error
"""
import os
import pathlib
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
import tensorflow as tf

import constants
import datasetloader
import model


if __name__ == '__main__':
    DATASET_SAVED_PATH:pathlib.Path = pathlib.Path(os.path.join(pathlib.Path(__file__).parents[0], 'ModelNet10'))
    checkpoint_path:pathlib.Path = pathlib.Path('./trained_model/checkpoints/cp-{epoch:04d}.ckpt')
    checkpoint_path.parents[0].mkdir(parents=True, exist_ok=True)
    checkpoint_dir:pathlib.Path = checkpoint_path.parents[0]


    """ Load the pointcloud dataset """
    train_points, test_points, train_labels, test_labels, class_map = datasetloader.processDataset(DATASET_SAVED_PATH, constants.UNIFORM_SAMPLE_COUNT)

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

    """ Save the weights using the 'checkpoint_path' format """
    model.save_weights(f"{checkpoint_path}".format(epoch=0))

    """ Compile the pointnet model """
    print('COMPILE THE POINTNET MODEL')
    model.compile(
                loss="sparse_categorical_crossentropy", 
                optimizer=tf.keras.optimizers.Adam(constants.LEARNING_RATE), 
                metrics=["sparse_categorical_accuracy"])
            
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print('REGISTERING ALL THE CALLBACKS ')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=f"{checkpoint_path}", verbose=1, save_weights_only=False, save_freq=10*constants.BATCH_SIZE),
        tensorboard_callback,
        # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=False)
    ]
    print('REGISTERED ALL THE CALLBACKS ')

    print('TRAINING STARTS ......')
    model.fit(train_dataset, epochs=30, validation_data=test_dataset, callbacks=callbacks)
    print('TRAINING ENDED......')