# RL for Quantum computing applications 

Install the packages in environment_working.yml file

Incase gym==0.21.0 has problem installing follow:

- pip install setuptools==65.5.0 pip==21  # gym 0.21 installation is broken with more recent versions
- pip install wheel==0.38.0

Using the parameter_values_NPois.py script change and store the parameters that wil be used for training and benchmarking.
This will be stored as a json file which is then loaded and used in main*.py scripts


Run the respective main file

The current example has two Poisson distributed request processes.

To evaluate the learned policy give its path in the main_evaluate_policy.py script. The parameters for evaluation are taken from the json file within that trained model. 


### Policies Implemented:
 - PPO based 
 - Always serve, never pump
 - Always serve, always pump 
 - Always serve, pump with a certain probability
 - Benchamrk 

Currently only two Poisson distributed request processes are used. 

