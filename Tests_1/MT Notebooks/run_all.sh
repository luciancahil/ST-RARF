#!/usr/bin/env bash

set -e  # stop immediately if any notebook fails

#jupyter nbconvert --to notebook --execute A-Nature2019-MT.ipynb   --inplace
#jupyter nbconvert --to notebook --execute B-DD2022--MT.ipynb --inplace
#jupyter nbconvert --to notebook --execute C-ACSCatal2024-MT.ipynb   --inplace
jupyter nbconvert --to notebook --execute D-ChemSci2024-MT.ipynb   --inplace
jupyter nbconvert --to notebook --execute E-JACS2023-MT.ipynb  --inplace
jupyter nbconvert --to notebook --execute F-Angew2024-MT.ipynb   --inplace
jupyter nbconvert --to notebook --execute H-Science2019-MT.ipynb --inplace







# Need to fix d. What's the problem?