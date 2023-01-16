import h5py
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataGenerator(Sequence):
    def __init__(self, folder_path, batch_size, input_shape, augment=False  ):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.augment = augment

        self.img_gen = ImageDataGenerator(
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True)

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.augment:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.input_shape))
        y = np.empty((self.batch_size, *self.input_shape))

        for i, ID in enumerate(list_IDs_temp):
            # Open the .h5 file
            with h5py.File(f'{self.folder_path}/{ID}', 'r') as hf:
                X[i,] = hf['raw'][:]
                y[i,] = hf['label'][:]
            
            # Preprocess and augment the data if desired
            if self.augment:
                X[i,], y[i,] = self.img_gen.random_transform(X[i,], y[i,])

        return X, y
