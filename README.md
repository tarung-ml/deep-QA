## Load dependencies: Activate conda virtual enviroment (also active on 
Azure machine)

* conda env create -f environment.yml (only once to create the env 
'deep')
* source activate deep

# Download data:
* sh code/get_started.sh
* python code/qa_data.py

# Initial run (also to perform a series of checks)
* python code/train.py

# checks on azure
* python gpu-test.py

