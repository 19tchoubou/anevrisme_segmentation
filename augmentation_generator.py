import os
import numpy as np
import h5py

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class AugmentedGenerator:
    '''
    Example Usage:
    >>> augmented_generator = AugmentedGenerator()
    >>> train_generator = augmented_generator.train_generator
    >>> validation_generator = augmented_generator.validation_generator
    >>> model.fit(train_generator, validation_data=validation_generator, epochs=10)
    '''
    def __init__(self, path='./challenge_dataset/', validation_split=0.2, batch_size=16, seed=1):
        self.path = path
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.seed = seed
        self.raw = None
        self.labels = None
        self._load_data()
        self.train_generator = self._get_train_generator()
        self.validation_generator = self._get_validation_generator()
    
    def _load_data(self):
        '''
        Loads the data from the .h5 files, might be better to use a generator here as well.
        '''
        data_exists = os.path.exists(self.path)

        if not data_exists:
            raise FileNotFoundError(f"Data folder not found at '{self.path}'")
        else:
            l_raw = []
            l_label = []
            for i in range(103): # might be better to make it cleaner
                f = h5py.File(self.path + f'scan_{i+1}.h5', 'r')
                l_raw.append(f['raw'])
                l_label.append(f['label'])
            self.raw, self.labels = np.array(l_raw), np.array(l_label)

    def _get_train_generator(self):
        '''
        Returns a generator for the training data with data augmentation.
        '''
        data_gen_params = dict(
            validation_split=self.validation_split,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            data_format='channels_first'
        )
        data_gen = ImageDataGenerator(**data_gen_params)
        
        return data_gen.flow(self.raw, self.labels, batch_size=self.batch_size, subset='training', seed=self.seed)


    def _get_validation_generator(self):
        '''
        Returns a generator for the validation data : no data augmentation.
        '''
        data_gen = ImageDataGenerator(
            validation_split=self.validation_split,
            data_format='channels_first')
        return data_gen.flow(self.raw, self.labels, batch_size=self.batch_size, subset='validation', seed=self.seed)
