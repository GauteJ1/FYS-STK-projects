The project with dependencies is installed by running
```
pip install .
```
in the ```/project_2``` folder. All following guides assumes you are located in this folder (paths are relative to this folder).

Current test status:

[![FYS-STK4155 Project "](https://github.com/GauteJ1/FYS-STK-projects/actions/workflows/test2.yml/badge.svg)](https://github.com/GauteJ1/FYS-STK-projects/actions/workflows/test2.yml)

All unit tests are located in ```src/tests.py```. They can be executed by:
```
pytest src/tests.py
```
Note that the initial installation of the project also includes the pytest package, hence no manual installation of this package will be needed in order to run this command.



All plots in the final report are generated in ```explorations/exploring_logreg.ipynb``` and ```explorations/exploring_nn.ipynb```. All numbers and results provided in the report can also be found in these files. Both these files are jupyter notebook files. Make sure to install the project as described before trying to run these files. The figures from these files are saved in ```figures/```.

The main code structure is located in the ```src/``` folder, while som intermediate code including weekly assignments are stored in ```weekly_assignments```.

