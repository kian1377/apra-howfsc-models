# apra-howfsc-models
This repo contains the telescope and coronagraph models used to test HOWFSC algorithms for a 6.5m aperture telescope.

In order to install this package into a python environment, download or clone this repository into a local directory. Then `cd` into this directory and run the command `python setup.py develop`. This will install this package into your activated python environment without creating a separate directory in that environemnet's site-packages directory. 

There is a separate package required for the HOWFSC algoirthms and that is the lina package found at https://github.com/uasal/lina. Here, algorithms such EFC, IEFC, PWP, and many other utility functions can be found. 

Currently, the main notebook to take a look at for an example of running EFC with PWP is efc_with_compact_model.ipynb. 





