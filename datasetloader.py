import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import tqdm
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
from matplotlib import pyplot as plt
import trimesh

import constants


def processDataset(datasetpath:pathlib.Path, sampling:int=constants.UNIFORM_SAMPLE_COUNT):
    train_points:list = []
    train_labels:list = []
    test_points:list = []
    test_labels:list = []
    class_map:object = {}
    directories = list(datasetpath.glob("[!README]*"))
    main_progress:tqdm.tqdm = tqdm.tqdm(enumerate(directories), desc="Parse Class: ", leave=True, total=len(directories))
    for i, directory in main_progress:

        training_files = list(pathlib.Path(os.path.join(directory, 'train')).glob('*.off'))
        testing_files = list(pathlib.Path(os.path.join(directory, 'test')).glob('*.off'))
        
        train_data_progress = tqdm.tqdm(enumerate(training_files), desc="Training Data: ", total=len(training_files), leave=True)
        test_data_progress = tqdm.tqdm(enumerate(testing_files), desc="Testing Data: ", total=len(testing_files), leave=True)
    
        main_progress.set_description(f'Parse class: {directory.stem}', refresh=True)

        for _, train_file in train_data_progress:
            train_data_progress.set_description(f'Class: {directory.stem}, Training File: {train_file.stem}', refresh=True)
            time.sleep(0.01)
        
        train_data_progress.set_description(f'Class: {directory.stem}, Total Training Files: {len(training_files)}', refresh=True)
        train_data_progress.update()
        main_progress.set_description(f'Parse class: {directory.stem}', refresh=True)

        for _, test_file in test_data_progress:
            test_data_progress.set_description(f'Class: {directory.stem}, Testing File: {test_file.stem}', refresh=True)
            time.sleep(0.01)
        
        test_data_progress.set_description(f'Class: {directory.stem}, Total Testing Files: {len(testing_files)}', refresh=True)
        test_data_progress.update()
        
        main_progress.set_description(f'Parse class: {directory.stem}', refresh=True)
        
        time.sleep(0.01)

def meshVisualization(meshpath:pathlib.Path):
    def addAxis(axis:matplotlib.axes.Axes, points:np.ndarray, *, color:str='red', title='3D plot'):
        axis.set_title(title)
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=color)
        axis.set_axis_off()
        return axis

    mesh:trimesh.Trimesh = trimesh.load(meshpath)
    points:np.ndarray = mesh.vertices
    #Using random sampling produce randomly distributed points along the surface of the mesh
    points_sampled:np.ndarray = mesh.sample(constants.UNIFORM_SAMPLE_COUNT)

    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(5, 5), subplot_kw=dict(projection='3d'))
    ax = addAxis(axis1, points, title = 'Original Vertices')
    ax = addAxis(axis2, points_sampled, color='blue', title = 'Uniformly Sampled')
    plt.show()


if __name__ == '__main__':
    tf.random.set_seed(constants.RANDOM_SEED)

    DATA_URL = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
    DATA_DIR = pathlib.Path(tf.keras.utils.get_file('ModelNet10.zip', 
                                                    DATA_URL, 
                                                    extract = True, ))
    DATA_DIR = pathlib.Path(os.path.join(DATA_DIR.parents[0], f'{DATA_DIR.stem}'))
    # mesh = meshVisualization(pathlib.Path(os.path.join(DATA_DIR, 'chair/train/chair_0001.off')))
    processDataset(DATA_DIR);