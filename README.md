# pacman-RL-BuildsAndTrainingScripts
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
16. Enter 'conda activate MLAgents20' into the prompt (this will change your MLAgents20 environment that you have created
17. Change directory to the 'ml-agents-release_20' directory from step 5
19. Enter 'pip install -e ml-agents'
20. Enter 'pip install -e ml-agents-envs'
21. Enter 'pip install matplotlib'
22. Enter 'pip install numpy'
23. Enter 'pip install pandas'
24. Enter 'pip install datetime'
25. Now that all packages are installed, navigate to the location of pacman-RL-BuildsAndTrainingScripts from step 1
26. Open pacman-RL-BuildsAndTrainingScripts in an IDE of your choice (All of our testing was done in Visual Studio Code)
27. In the 'TrainingScripts' directory locate the file 'pacmanDQCNN.py'
28. Open 'pacmanDQCNN.py' and run this file to begin training

