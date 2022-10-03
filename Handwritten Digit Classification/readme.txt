Neural Computing Coursework

Please find contents of the zipped folder, instructions for testing and library dependencies below

1. Contents of folder

PDF/HTML Files - These files correspond to the code transformed to HTML files as requested in the coursework documentation

	- NC_CW_MLP_Train.html
	- NC_CW_MLP_Test.html
	- NC_CW_SVM_Train.html
	- NC_CW_SVM_Test.html

Ipynb Files - These files correspond to the code itself as Jupyter notebooks

	- NC_CW_MLP_Train.ipynb
	- NC_CW_MLP_Test.ipynb: PLEASE RUN THIS FOR TESTING FOR MULTILAYER PERCEPTRON
	- NC_CW_SVM_Train.ipynb
	- NC_CW_SVM_Test.ipynb: PLEASE RUN THIS FOR TESTING FOR SUPPORT VECTOR MACHINES

NOTE: Given that this code was prepared in Google Colab, there exist certian cells in the test scripts that are not required to run.
Explicit warnings have been added to the top of these cells. PLEASE DO NOT RUN THEM!

Pickle Files - These files contain the saved models and will be loaded in the test files above

	- MLP_trained.pkl
	- SVM_trained.pkl

Data folder - This contains the training and test sets used for model training. This data will be read from the test scripts during testing


2. Library dependencies

numpy Version 1.20.1
pandas Version 1.2.4
matplotlib Version 3.3.4
seaborn Version 0.11.1
torch Version 1.11.0
sklearn Version 0.24.1
skorch Version 0.11.0
