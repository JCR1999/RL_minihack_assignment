# Structure of SUBMISSION FOLDER:
   # DQN - folder that contains DQN code, videos, plot code, and text files
    minihack_assignment_dqn.py - the main code for the DQN agent. Run this file with python3 minihack_assignment_dqn.py
    starter_code - this folder contains the DQN code and relavent classes, called from minihack_assignment_dqn.py
    videos - contains the video of the agent playing the game.
   # Reinforce
    main.py - this file contains the hyperparameters given to reinforce.

   In main.py, make a Reinforce object and train a policy using the train method. The policy will be saved to the PATH that you   pass in with the hyperparameters. If a policy exists at that path, use the load_policy method to load it. Test a policy using the test_policy method. This will run the policy for a number of episodes and output the average reward. Uncomment the (#render env) code to render the environment on the console at each step.

   To see the effects of the action value, epsilon and negative reward for unchanged state code, uncomment the sections one at a time in the reinforce.py file
