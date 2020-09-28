# PDEKoopman
## Using neural networks to learn linearizing transformations for PDEs

This code is for the paper ["Deep Learning Models for Global Coordinate Transformations that Linearise PDEs"](https://www.cambridge.org/core/journals/european-journal-of-applied-mathematics/article/deep-learning-models-for-global-coordinate-transformations-that-linearise-pdes/4C3252EA5D681D07D933AD31EE539192) by Craig Gin, Bethany Lusch, Steven L. Brunton, and J. Nathan Kutz. All of the code is written for Python 2 and Tensorflow 1.

Our recommendation is to only use this code to verify the results of the above paper. If you are interested in adapting the code and using it for a problem you are working on, you may want to wait. We are currently working on updating the code to run with Python 3 and Tensorflow 2 and as part of this update will be making efforts to make the code much more user-friendly. We will update this repository with a link when that is available. As of Sept. 28, 2020, the new version of the code is just in need of some final testing so it should be available soon.

To run the code:

1. Clone the repository.
2. In the data directory, recreate the desired datasets. The data sets used in the paper are created with the files Heat_Eqn_exp29_data.m, Burgers_Eqn_exp28_data.m, Burgers_Eqn_exp30_data.m, Burgers_Eqn_exp32_data.m, KS_Eqn_exp4_data.py, KS_Eqn_exp5_data.py, KS_Eqn_exp6_data.py, and KS_Eqn_exp7_data.py. If you create data using one of the MATLAB .m files, you will then need to convert the resulting csv files to .npy files which can be done with the script csv_to_npy.py.
3. In the main directory, run the desired experiment files. As an example, Burgers_Experiment_28rr.py will train 20 neural networks with randomly chosen learning rates and initializations each for 20 minutes. It will create a directory called Burgers_exp28rr and store the networks and losses. You can then run the file Burgers_Experiment28rr_restore.py to restore the network with the smallest validation loss and continue training the network until convergence.
4. The Jupyter notebooks in the main directory (Heat_PostProcess.ipynb, Burgers_Postprocess.ipynb, KS_Postprocess.ipynb) can be used to examine the results and create figures like the ones in the paper. 

All of the results from the paper are already in the repository so you can exactly recreate the paper figures by running the Jupter notebooks "FigureXX.ipynb".
