Welcome to the repository for **Iterative Hard Thresholding (IHT)** applied to **Accelerated Gradient Descent (AGD)**. 


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/2560px-PyTorch_logo_black.svg.png" width="128"/>

# Instructions

## The notebook :notebook:
Download the *main.ipynb* file for trying out experiments.
You can run all of the cells and main.ipynb will automatically import the supporting code found in this repository, as well as some other external imports. Alternatively you can also run the notebook in colab with this link:
https://colab.research.google.com/drive/1ufxJD_hi5soLwSKcuO1dz0rO_I3P_FU5?usp=sharing

## Experiments :alembic:
You can adjust the variables for each type of optimizer (SGD, AGD, IHT-AGD, etc.) in the 'Setups' section. Furthernore, it is also possible to inherit from these optimizer classes to make new optimizers and customize what variables to track

## Real-Time Tracking :chart_with_upwards_trend:
As an MLOps service, this project uses Neptune.ai, the link for which is:
https://app.neptune.ai/o/dimitri-kachler-workspace/org/sanity-MNIST/runs/details?viewId=standard-view&detailsTab=dashboard&dashboardId=9c26ef22-c91b-462a-bddf-babfd83a481f&shortId=SAN-489

## Miscellaneous Notes :grey_exclamation:
One can also specify an experiment to visualize, but this requires the experiment ID which should be printed as output. The ID is in the format of "SAN-???".


