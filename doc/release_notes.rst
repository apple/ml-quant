=============
Release Notes
=============

Current Release
===============

v.0.2.0 (2020/06/15)
--------------------------------------

 * Improved documentation
 * Module re-organization: move modules from `common` to `utils`
 * Fix moving average bugs
 * Use original loss function instead of kd loss function for eval
 * LeNet quantization bugfixes
 * Remove unneeded data augmentation from data loader

v.0.1.0 (2020/03/30)
--------------------------------------

 * Initial release of the library
 * Support for the following quantization methods: least squares 1-bit (ls-1), 2-bits (ls-2), ternary (ls-T), and greedy foldable (gf)
 * Dataset loaders for MNIST, CIFAR-10, CIFAR-100, ImageNet
 * Quantized module for ``nn.Conv2d``
 * LeNet and ResNet (regular block and XNOR block variants) models
 * Code required for running training and inference
 * Support for training with a teacher
 * Support for using moving average during inference to avoid re-computing scalars

Known Issues
------------

 * If you installed all of the dependencies following the instructions, but get TensorBoard not found, try deactivating the virtualenv and re-activating it.
