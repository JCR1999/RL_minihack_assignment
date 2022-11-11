reinforce.py - the main code for the REINFORCE policy.

main.py - this file contains the hyperparameters given to reinforce. 
In main.py, make a Reinforce object and train a policy using the train method. The policy will be saved to the PATH that you pass in with the hyperparameters. 
If a policy exists at that path, use the load_policy method to load it.
Test a policy using the test_policy method. This will run the policy for a number of episodes and output the average reward. Uncomment the (#render env) code to
render the environment on the console at each step.

To see the effects of the action value, epsilon and negative reward for unchanged state code, uncomment the sections one at a time in the reinforce.py file
