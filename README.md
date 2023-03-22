# pacman-RL-BuildsAndTrainingScripts
Feel free to reach out if you would like a group member to demonstrate running the code.

Builds and training scripts for deep reinforcement learning pacman AI
1. Download the pacman-RL-BuildsAndTrainingScripts to a directory of your choosing
2. Navigate to: https://github.com/Unity-Technologies/ml-agents
3. Scroll down to 'Releases & Documentation' section
4. Locate the release table and locate Release 20 from Nov,21,2022
5. Click the download button and unzip 'ml-agents-release_20' to a directory of your choosing
6. Navigate to: https://www.anaconda.com/products/distribution
7. Click the download button to download the installer
8. Run the installer to install anaconda
9. Once installed, Run the Anaconda Navigator application
10. Select 'Environments' on the left hand side of the application
11. Click the 'Create' button in the lower left hand corner to create an environment
12. In the 'Name' dialog box enter: MLAgents20
13. In the Version Selection dropdown box select Python Version: 3.9.16
14. Press the create button
15. Open an 'Anaconda Prompt' from the Windows start menu
16. Enter 'conda activate MLAgents20' into the prompt (this will change your MLAgents20 environment that you have created)
17. Change directory to the 'ml-agents-release_20' directory from step 5
19. Enter 'pip install -e ml-agents'
20. Enter 'pip install -e ml-agents-envs'
21. Enter 'pip install matplotlib'
22. Enter 'pip install numpy'
23. Enter 'pip install pandas'
24. Enter 'pip install datetime'
25. Enter 'pip install protobuf==3.19.4'
26. Now that all packages are installed, navigate to the location of pacman-RL-BuildsAndTrainingScripts from step 1
27. Open pacman-RL-BuildsAndTrainingScripts in an IDE of your choice (All of our testing was done in Visual Studio Code)
28. Ensure the interpreter chosen is the conda environment created in earlier steps
29. In the 'TrainingScripts' directory locate the file 'pacmanDQCNN.py'
30. Open 'pacmanDQCNN.py' and run this file to begin training

After running pacmanDQCNN.py is complete, loss vs episode, loss vs epsilon, meanQ vs episode, reward vs episode and epsilon vs episode graphs will be stored in TrainingScripts/Logs in a folder with the run time and date of the training.


PacmanDQCNN.py is currently setup to run our MLP model as it can be run in a reasonable amount of time for demonstration/grading purposes. 
The environment can also run the CNN architecture that is discussed in our slides and paper by:
    1. Changing FIRST_LAYER_FULLY_CONNECTED variable to False on line 82
    2. Comment out the current training/target layers lists on line 274-275
    3. Un-comment the training/target layers list on lines 270-271
    (Runtimes for CNN training can take hours)

A trained model can then be tested in the 'modelTesting.py' file.
Weights are periodically stored as .npy files and can be loaded into modelTesting.py by editing the path of the toLoad variable
The testModelLayers list must also accurately reflect the same architecture that the weights were trained on.
Weight .npy files are stored in TrainingScripts/NPY_Files

