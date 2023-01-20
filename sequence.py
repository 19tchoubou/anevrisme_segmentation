import numpy as np
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, raw_data, labels, batch_size, input_shape, output_channels=1, augmentator=None, shuffle=True):
        """Custom dataset.

        Args:
            folder_path (_type_): path to the folder where scans are stored.
            batch_size (_type_): batchsize for training.
            input_shape (_type_): shape of one scan after opening it.
            augmentator (_type_): Volumentation augmentator.
            shuffle (bool, optional): shuffle dataset between epochs. Defaults to True.
        """
        self.raw_data = raw_data
        self.labels = labels
        
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_channels = output_channels
        
        self.shuffle = shuffle 
        self.augmentator = augmentator
        
        self.N = len(raw_data) # num samples in one epoch
        self.indexes = np.arange(self.N) # [0, N-1] : position in batch

    def __len__(self):
        """DataGenerator is an iterator, so this function returns the number of batches.
        We take floor instead of ceil to avoid issues with half a possible filled last batch

        Returns:
            int: number of batches.
        """
        return int(np.floor(self.N / self.batch_size))

    def __getitem__(self, index):
        """Get i-th batch. Returns the element i*batchsize to (i+1)*batchsize

        Args:
            index (int): index of the batch

        Returns:
            np.array: the batch
        """
        # indexes to take for the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes) # inplace method

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, *self.input_shape))
        Y = np.empty((self.batch_size, *self.input_shape))

        for i, ID in enumerate(indexes):
            x = self.raw_data[ID]
            y = self.labels[ID]
        
            # Preprocess and augment the data if desired
            if self.augmentator is not None:
                data = {'image': x, 'mask': y}
                aug_data = self.augmentator(**data)
                x, y = aug_data['image'], aug_data['mask']
                
            X[i,], Y[i,] = x, y

        # we add an extra dimension for channels bc 3D models expect one
        # X = np.stack([X, X, X], axis=-1) # dirty way to go RGB from grayscale
        X = np.expand_dims(X, axis=-1) 
        Y = np.expand_dims(Y, axis=-1) 
        
        return X, Y