# ophiuchus-al26
Code to infer properties of Upper Sco based on Measurements of 26Al decay via gamma-rays

This code implements the inference framework in Forbes, Alves, and Lin (2021).

Most of the plots related to this inference can be reproduced by running 

python3 -u aluminum_inference.py

though this will take a while (many hours), and many of the plots are non-deterministic, i.e. they are based on a small random subsample of the provided posterior samples.

The code requires, in addition to a number of python modules, access to the Geneva Stellar Tracks, available at  https://obswww.unige.ch/Research/evol/tables_grids2011/tablesZ014.tgz

The untarred contents of this file are expected by the code in a subdirectory called ekstrom_geneva.

The python libraries expected include numpy, schwimmbad, matplotlib, scipy, and dynesty, all of which are pip-installable.
