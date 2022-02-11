# Flexible time series forecasting
Codes for flexible time series multivariate multi-step forecasting using machine learning

A repository for this medium article:

<a target="_blank" href="https://github-readme-medium-recent-article.vercel.app/medium/@shahrzadhadian/0"><img src="https://github-readme-medium-recent-article.vercel.app/medium/@shahrzadhadian/0" alt="Recent article #1"></a>

The [window generator class](./data/create_datasets.py) is a generalisation of the sliding window technique from [Tensorflow blog](https://www.tensorflow.org/tutorials/structured_data/time_series)
in Numpy with more flexible input/output features. It formulates the series as a 3d supervised learning dataset.

The [neural network](./models/models_neural_networks.py) module is a collection of models that use the window generator as input to produce flexible input/output size
forecasting. Besides the models the recursive and direct classes in the module are different strategies of producing multi-step forecasts. The former inherits from the window generator
and the latter instantiates multiple window generators to produce multi-steps in the future.

The [classic ML](./models/models_classic_ml.py) module contains some auxiliary functions that facilitate the usage of any vector output (two dimensional) classic ML model with the 3d data generated from the window generator.
It also contains the subtly changed recursive and direct classes adopted to work with the classic ML models.

The usage of the codes can be seen in the [Jupyter notebooks](./notebooks): Running the 'Jupyter_analytics' on the datasets in [datasets](./datasets) will produce all the images
of the [medium article](https://medium.com/p/6e967f3c1e6b/edit). The 'future_forecast' shows how the codes can be used to produce forecast for an unseen future.

# A note on the environment.yml
Ordinarily you can clone the repository, Cd into it, replicate my coding environment by:
```
conda env create -f environment.yml
```
but given I am using Apple arm and installing Tensorflow was less than trivial, the environment.yml file may not be reproduce my environment exactly.
If you are using apple arm on Mac with Big Sur, now you can follow [these steps](https://github.com/apple/tensorflow_macos) to install Tensorflow optimized for apple silicon. For the rest of the installed libraries and packages in the environmen.yml file should work fine.
Although I have noticed faster training speed, there are still bugs and issues with Tensorflow 2.4 on arm.

Training some of the networks on this repository may be slow but you don't really need GPUs. If you wish to use apple silicon GPU power after installing Tensorflow do the following:
```
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompte.set_mlc_device(device_name='gpu')
```



