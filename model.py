import numpy as np
import keras
import tensorflow as tf

import constants

def convolutionBatchNormalization(layer:keras.engine.keras_tensor.KerasTensor, filters:int)-> keras.engine.keras_tensor.KerasTensor:
    layer = tf.keras.layers.Conv1D(filters, kernel_size=1, padding="valid")(layer)
    layer = tf.keras.layers.BatchNormalization(momentum=0.0)(layer)
    return tf.keras.layers.Activation('relu')(layer)

def denseBatchNormalization(layer:keras.engine.keras_tensor.KerasTensor, filters:int)->keras.engine.keras_tensor.KerasTensor:
    layer = tf.keras.layers.Dense(filters)(layer)
    layer = tf.keras.layers.BatchNormalization(momentum=0.0)(layer)
    return tf.keras.layers.Activation('relu')(layer)

def transformerNet(inputs:keras.engine.keras_tensor.KerasTensor, num_features:int) -> keras.engine.keras_tensor.KerasTensor:
    # Create the initial set of bias weights
    bias:tf.keras.initializers.Constant = tf.keras.initializers.Constant(np.eye(num_features).flatten())
    reg:OrthogonalRegularizer = OrthogonalRegularizer(num_features)

    x = convolutionBatchNormalization(inputs, 32)
    x = convolutionBatchNormalization(x, 64)
    x = convolutionBatchNormalization(x, 512)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    x = denseBatchNormalization(x, 256)
    x = denseBatchNormalization(x, 128)
    x = tf.keras.layers.Dense(num_features**2, kernel_initializer='zeros', bias_initializer=bias, activity_regularizer=reg) (x)
    featureTransform = tf.keras.layers.Reshape((num_features, num_features))(x)

    return tf.keras.layers.Dot(axes=(2, 1))([inputs, featureTransform])

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features:int, l2reg:float = 0.001, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)
    
    def __call__(self, layer:keras.engine.keras_tensor.KerasTensor) -> keras.engine.keras_tensor.KerasTensor:
        layer = tf.reshape(layer, (-1, self.num_features, self.num_features))
        layerdotlayertranspose = tf.tensordot(layer, layer, axes=(2, 2))
        layerdotlayertranspose = tf.reshape(layerdotlayertranspose, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(layerdotlayertranspose - self.eye))# (A.A^T) - I = 0
    
    def get_config(self):
        # config = super().get_config().copy()
        # config.update({
        #     'num_features': self.num_features,
        #     'l2reg': self.l2reg
        # })
        config = {
            'num_features': self.num_features,
            'l2reg': self.l2reg
            }

        return config



def createPointNetModel(num_points:int = constants.UNIFORM_SAMPLE_COUNT, num_classes:int = constants.NUM_CLASSES)->tf.keras.Model:
    inputs = keras.Input(shape=(num_points, 3))

    x = transformerNet(inputs, 3)
    x = convolutionBatchNormalization(x, 32)
    x = convolutionBatchNormalization(x, 32)
    x = transformerNet(x, 32)
    x = convolutionBatchNormalization(x, 32)
    x = convolutionBatchNormalization(x, 64)
    x = convolutionBatchNormalization(x, 512)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = denseBatchNormalization(x, 256)
    x = keras.layers.Dropout(0.3)(x)
    x = denseBatchNormalization(x, 128)
    x = keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='pointnet')
    return model

if __name__ == '__main__':
    model = createPointNetModel(constants.UNIFORM_SAMPLE_COUNT, constants.NUM_CLASSES)
    model.summary()