# 3D intracranial aneurysm semantic segmentation using 3D-Unet

Here are the useful pieces of our work you could be interested in :
- Import/Export a dataset from/into .h5 files
- Tensorflow generator for 3D images + 3D data augmentation 
- 3D semantic segmentation training pipeline
- Viz tools for 3D voxel segmented objects

## File organization

In this project, we use various flavors of 3D-Unet to segment intracranial aneurysms in MRI scans.

The dataset is made of 103 scans, of shape (103, 64, 192, 192). Their masks are binary segmentations of the aneurysms. Scans are approximatively centered on the biggest aneurysm of the volume.

To explore your dataset, run ```explore_data.ipynb```.

To make predictions, you place your dataset to segment in a ```./to_predict/``` folder and follow the instructions of ```make_predictions.ipynb```. A pretrained model is required for the prediction task.

To train a model on your own, run the ```train_3Dunet.ipynb``` notebook and custom your training. We use both custom architectures and pre-built ones (using segmentation-models-3d to try various encoder backbones), trained with augmented datsets (using volumentations-3d data augmentation tools).

![](img/scan_10_finetuned_crop.gif.gif)

![](img/scan_8_finetuned_crop.gif.gif)