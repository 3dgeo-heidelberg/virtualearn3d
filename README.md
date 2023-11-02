# virtualearn3d

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/virtualearn3d/badge/?version=latest)](https://virtualearn3d.readthedocs.io/en/latest/?badge=latest)


Welcome to the VirtuaLearn3D (VL3D) framework for artificial intelligence
applied to point-wise tasks in 3D point clouds.


## Install

### Machine learning install

Since the VL3D framework is a Python-based software, the installation is quite
simple. It consists of three steps:


1. Clone the repository.

    ```bash
    git clone https://github.com/3dgeo-heidelberg/virtualearn3d
    ```

2. Change the working directory to the framework's folder.
 
    ```bash
    cd virtualearn3d
    ```

3. Install the requirements.

    ```bash
    pip -r requirements.txt
    ```

### Deep learning install

If you are interested in deep learning, you will need some extra steps
to be able to use the GPU with TensorFlow+Keras. You can see the
[TensorFlow documentation](https://www.tensorflow.org/install/pip)
on how to meet the hardware requirements when installing with
pip. Below you can find a summary of the steps that work in the general case:

4. Check that your graphic card supports CUDA. You can check it 
    [here](https://developer.nvidia.com/cuda-gpus).
 
5. Install the drivers of your graphic card.

6. Install CUDA. See either the
[Linux documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/contents.html)
or the
[Windows documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
on how to install CUDA.

7. Install cuDNN. See the
[cuDNN installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).





## Usage


### Pipelines

In the VL3D framework, you can define pipelines with many components that
handle the computation of the different steps involved in a typical artificial
intelligence workflow. These components can be used to compute point-wise
features, train models, predict with trained models, evaluate the predictions,
write the results, apply data transformations, define imputation strategies,
and automatic hyperparameter tuning.

For example, a pipeline could start computing some
geometric features for a point-wise characterization, then train a model,
evaluate its performance with k-folding, and export it as a predictive
pipeline that can later be used to classify other point clouds. Pipelines
are specified in a JSON file, and they can be executed with the command below:


```bash
python vl3d.py --pipeline pipeline_spec.json
```

You can find information about pipelines and the many available components
in our [documentation](). As the first step, we recommend understanding how pipelines
work, i.e., reading the
[documentation on pipelines]().





### Deep learning test

If you are interested in the deep learning components of the VL3D framework,
check that your TensorFlow+Keras installation is correctly linked
to the GPU. Otherwise, deep learning will be infeasible. To check this,
run the following command:



```bash
python vl3d.py --test
```

You should see that all tests are passed. Especially the
**Keras, TensorFlow, GPU test** and the **Receptive field test**.
