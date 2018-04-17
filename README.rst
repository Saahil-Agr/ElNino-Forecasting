climateAI-forecast
##################

Software to develop deep learning models to make seasonal forecasts of climate phenomena.

For example a model could be developed to predict an ENSO index - `Nino3.4 <https://iridl.ldeo.columbia.edu/maproom/ENSO/Diagnostics.html>`_ - at a 1 year lead time. To train the model we use thousands of years of simulated data from climate models (GCMs) to establish relationships between patterns in time of global climate variables and the state of Nino3.4. 

Installation
------------

Python 3.6

.. code-block:: bash

    $ pip3 install -r requirements.txt
  
Data Preprocessing
------------------

The script ``prepreprocess_gcm.py`` performs three data preprocessing tasks. 

1. regridding GCM to a common spatial grid and computing anomalies
2. computing a response variable time series by taking the mean of a spatial subset for a specific variable
3. writing individual image files for each time step

The script requires a specific file structure to work and has several settings that can be changed. See the docstrings and comments of the script for details.
