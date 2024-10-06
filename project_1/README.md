The project with dependencies is installed by running
```
pip install .
```
in the ```/project_1``` folder. All following guides assumes you are located in this folder (paths are relative to this folder).

Current test status:

[![FYS-STK4155 Project 1](https://github.com/GauteJ1/FYS-STK-projects/actions/workflows/test1.yml/badge.svg)](https://github.com/GauteJ1/FYS-STK-projects/actions/workflows/test1.yml)

All unit tests are located in ```src/tests.py```. They can be executed by:
```
pytest src/tests.py
```
Note that the initial installation of the project also includes the pytest package, hence no manual installation of this package will be needed in order to run this command.

All plots in the final report are generated in ```src/plots_in_report.ipynb```, and further plots added in the appendix are generated in ```src/plots_in_appendix.ipynb```. Both these files are jupyter notebook files. Make sure to install the project as described before trying to run these files. The figures from these files are saved in ```figures/figures_in_report``` and ```figures/figures_in_appendix``` respectively.

Finally, the grid search is performed in ```src/grid_search.ipynb```. After performing the grid search (this takes a while) the optimal hyper-parameters are saved in ```src/best_params.json```, which is used when plotting.