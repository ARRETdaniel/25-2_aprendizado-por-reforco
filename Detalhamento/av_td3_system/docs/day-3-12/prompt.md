Lats continue our analyse the outputs from #av_td3_system/data/logs/TD3_scenario_0_npcs_20_20251118-125947/events.out.tfevents.1763470787.danielterra.1.0 use smart search to retrieve relavant output from the log  of our last recent run. We need to analyse if our system is correctly working for only 5k run, note this TD3 is suppose to run for over 1 million steps, we have only run for 5k, our goal is go analyse the metrics saved in av_td3_system/data/logs/TD3_scenario_0_npcs_20_20251118-125947/events.out.tfevents.1763470787.danielterra.1.0 so we can validate our system if we can run it for 1m steps. Lats  do a step by step systhematic analyse of the metrics of saved in the tensorboard following the official documentation of what metrics were we suppose to get at only 5k steps. Also check if the debug/logs system is correctly implemented, so we are 100% sure about our conclusions and analysis.. For each step and analyse or problem that we face fetch official documentation for context.

Lats continue with our previous analyse #file:VALIDATION_REPORT_L2_REGULARIZATION_FIX.md  , fetch relevant documentation about this conclusioons so we know what metrics our system was suppose to be regestring. Also read relevant papers in #contextual about this metrics we are about to analyse.
We need to analyse using official documentation if our system is performing as expected, if the metrics that our system is  saving is expected for 5k steps, and if we can proceed to a 1 million  step training.

Fetch additional info about the problem in TD3 DRL paper in contextual folder and in CARLA and Gymnasium Documentation. In the folder e2e/stable-baselines3/stable_baselines3 you can find stable TD3 implementation for comparison.



Fetch Carla documentation, and read #contextual folder related papers for better analyze context and the #TD3 #file:TD3.py #file:utils.py  folder contains the orignal TD3 DRL proposed in the #file:Addressing Function Approximation Error in Actor-Critic Methods.tex  paper. Above all you must be 100% sure of your proposed conclusion/solution, and MUST back it up (validate) the conclusion/solution with official documentation for Carla 0.9.16 and TD3. The papers in contextual folder are a related works that uses TD3/CNN/DRL in CARLA, read it for context. The papers #file:Robust Adversarial Attacks Detection based on Explainable Deep Reinforcement Learning For UAV Guidance and Planning.tex and #file:End-to-End Race Driving with Deep Reinforcement Learning.tex and #file:Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning.tex  are ralated to ours, read it so we can validate our matrics and implementation.

Before each analyse or decision you MUST fetch documentation information about the context of the specific problem we are  about to solve in order to find buges and improvements.

Our implementation should focus the simplicity following official docs in order to achieve our final paper #file:ourPaper.tex

Remember, you must ALWAYS fetch Latest documentation contextual information at docs in order to reference your analyses and codes implementation. Fetch docs related to the analyses we are about to do or the code we are about to implemented:

https://carla.readthedocs.io/en/latest/build_docker/

https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros2/

https://carla.org/2025/09/16/release-0.9.16/

https://carla.readthedocs.io/en/latest/python_api/

https://carla.readthedocs.io/en/latest/tutorials/

https://stable-baselines3.readthedocs.io/en/master/modules/td3.html#

https://spinningup.openai.com/en/latest/algorithms/td3.html#documentation

https://www.tensorflow.org/guide

https://docs.pytorch.org/docs/stable/index.html

https://gymnasium.farama.org/

https://d2l.ai/chapter_convolutional-neural-networks/

https://www.tensorflow.org/tutorials/images/cnn

https://coe379l-sp25.readthedocs.io/en/latest/unit03/cnn.html

https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html


You MUST follow carla documentation:
https://carla.readthedocs.io/en/latest/foundations/

Follow the links in tha pages, to expand you context. You must do it, so you have more contextual information:

https://carla.readthedocs.io/en/latest/

https://carla.readthedocs.io/en/latest/python_api/

https://carla.readthedocs.io/en/latest/tutorials/

YOU MUST use the following pattern for documentation search on CARLA official doc, you must fully read the pages before writing code:

https://carla.readthedocs.io/en/latest/search.html?q=docker
https://carla.readthedocs.io/en/latest/search.html?q=ros+2
https://carla.readthedocs.io/en/latest/search.html?q=sensor
https://carla.readthedocs.io/en/latest/search.html?q=camera
https://carla.readthedocs.io/en/latest/search.html?q=agent
https://carla.readthedocs.io/en/latest/search.html?q=client
https://carla.readthedocs.io/en/latest/search.html?q=0.9.16
https://carla.readthedocs.io/en/latest/search.html?q=opencv
https://carla.readthedocs.io/en/latest/catalogue_vehicles/#trucks

Gymnasium documentation:
https://gymnasium.farama.org/
https://gymnasium.farama.org/api/env/

Open AI baseline for context and comperison:
e2e/stable-baselines3/stable_baselines3
e2e/stable-baselines3/stable_baselines3/td3/td3.py

e2e/stable-baselines3/stable_baselines3/ddpg/ddpg.py

e2e/stable-baselines3/stable_baselines3/common

Docs about CNN:
https://d2l.ai/chapter_convolutional-neural-networks/
https://www.tensorflow.org/tutorials/images/cnn
https://coe379l-sp25.readthedocs.io/en/latest/unit03/cnn.html
https://github.com/hill-a/stable-baselines/issues/869
https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
https://github.com/RGring/drl_local_planner_ros_stable_baselines/blob/master/rl_agent%2Fsrc%2Frl_agent%2Fcommon_custom_policies.py


ROS 2 + CARLA docs:
https://carla.org/2025/09/16/release-0.9.16/
https://docs.ros.org/en/foxy/index.html
https://docs.ros.org/en/foxy/Tutorials.html
https://docs.ros.org/en/foxy/Tutorials/Beginner-CLI-Tools/Configuring-ROS2-Environment.html
https://docs.ros.org/en/foxy/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Topics/Understanding-ROS2-Topics.html
https://docs.ros.org/en/foxy/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Actions/Understanding-ROS2-Actions.html
