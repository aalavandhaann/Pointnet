import os
import shutil
import time
import pathlib
import tqdm
import numpy as np
import scipy.io as sio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
from matplotlib import pyplot as plt
import trimesh
import constants

def loadModelNet10(save_path:pathlib.Path):
    DATA_URL:str = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
    DATA_DIR:pathlib.Path = pathlib.Path(tf.keras.utils.get_file('ModelNet10.zip', 
                                                    DATA_URL, 
                                                    extract = True, ))
    DATA_DIR = pathlib.Path(os.path.join(DATA_DIR.parents[0], f'{DATA_DIR.stem}'))

    shutil.move(DATA_DIR, save_path)

    return save_path


def augmentation(points3d:np.ndarray, label:str) -> tuple:
    #Jitter the 3d points by a distance of 0.01 i.e from -0.005 to 0.005
    points3d += tf.random.uniform(points3d.shape, -0.005, 0.005, dtype=tf.float64)
    #shuffle the points i.e shuffle the ordering of point indices
    points3d = tf.random.shuffle(points3d)
    return points3d, label

def processDataset(datasetpath:pathlib.Path = pathlib.Path('./ModelNet10'), sampling:int = constants.UNIFORM_SAMPLE_COUNT) -> tuple:
    mat_path:pathlib.Path = pathlib.Path(os.path.join(datasetpath, 'dataset.mat'))

    if(not mat_path.exists()):
        if(not datasetpath.exists()):
            loadModelNet10(datasetpath)

        train_points:list = []
        train_labels:list = []
        test_points:list = []
        test_labels:list = []
        class_map:object = {}
        directories = list(datasetpath.glob("[!README]*"))
        main_progress:tqdm.tqdm = tqdm.tqdm(enumerate(directories), desc="Parse Class: ", leave=False, total=len(directories))
        for classLabelIndex, directory in main_progress:

            training_files:list = list(pathlib.Path(os.path.join(directory, 'train')).glob('*.off'))
            testing_files:list = list(pathlib.Path(os.path.join(directory, 'test')).glob('*.off'))
            
            train_data_progress:tqdm.tqdm = tqdm.tqdm(enumerate(training_files), desc="Training Data: ", total=len(training_files), leave=False)
            test_data_progress:tqdm.tqdm = tqdm.tqdm(enumerate(testing_files), desc="Testing Data: ", total=len(testing_files), leave=False)
        
            main_progress.set_description(f'Parse class: {directory.stem}', refresh=True)

            for _, train_file in train_data_progress:
                train_data_progress.set_description(f'Class: {directory.stem}, Training File: {train_file.stem}', refresh=True)
                train_points.append(trimesh.load(train_file).sample(sampling))
                train_labels.append(classLabelIndex)
            
            train_data_progress.set_description(f'Class: {directory.stem}, Total Training Files: {len(training_files)}', refresh=True)
            train_data_progress.update()
            main_progress.set_description(f'Parse class: {directory.stem}', refresh=True)

            class_map[classLabelIndex] = directory.stem

            for _, test_file in test_data_progress:
                test_data_progress.set_description(f'Class: {directory.stem}, Testing File: {test_file.stem}', refresh=True)
                test_points.append(trimesh.load(test_file).sample(sampling))
                test_labels.append(classLabelIndex)
            
            test_data_progress.set_description(f'Class: {directory.stem}, Total Testing Files: {len(testing_files)}', refresh=True)
            test_data_progress.update()
            
            main_progress.set_description(f'Parse class: {directory.stem}', refresh=True)
        
        sio_data = {
                    'train_points': np.array(train_points), 
                    'test_points': np.array(test_points), 
                    'train_labels': np.array(train_labels), 
                    'test_labels': np.array(test_labels), 
                    'class_map': class_map}
                    
        sio.savemat(mat_path, sio_data)
    
    mat_data = sio.loadmat(mat_path)
    train_points = mat_data['train_points']
    test_points = mat_data['test_points']
    train_labels = mat_data['train_labels']
    test_labels = mat_data['test_labels']
    class_map = mat_data['class_map']
    return (np.array(train_points), np.array(test_points), np.array(train_labels), np.array(test_labels), class_map)

def meshVisualization(meshpath:pathlib.Path) -> None: 
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

    DATASET_SAVED_PATH:pathlib.Path = pathlib.Path(os.path.join(pathlib.Path(__file__).parents[0], 'ModelNet10'))

    # if(not DATASET_SAVED_PATH.exists()):
    #     DATA_URL:str = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
    #     DATA_DIR:pathlib.Path = pathlib.Path(tf.keras.utils.get_file('ModelNet10.zip', 
    #                                                     DATA_URL, 
    #                                                     extract = True, ))
    #     DATA_DIR = pathlib.Path(os.path.join(DATA_DIR.parents[0], f'{DATA_DIR.stem}'))

    # mesh = meshVisualization(pathlib.Path(os.path.join(DATA_DIR, 'chair/train/chair_0001.off')))
    train_points, test_points, train_labels, test_labels, class_map = processDataset(DATASET_SAVED_PATH, constants.UNIFORM_SAMPLE_COUNT)

    print('='*40)
    print(train_points.shape)
    print(test_points.shape)
    print(train_labels.shape)
    print(test_labels.shape)
    print(class_map)