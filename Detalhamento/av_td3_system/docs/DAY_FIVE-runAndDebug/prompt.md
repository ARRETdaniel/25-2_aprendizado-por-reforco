Lats continue our analyse the outputs from #file:DEBUG_validation_20251105_194845.log  (av_td3_system/docs/DAY_FIVE-runAndDebug/DEBUG_validation_20251105_194845.log) use smart search to retrieve relavant output from the log  of our last recent run. We need to analyse if our system is getting and outputing and transforming the data as expected. Lats start by the debuging the data flow of our system as shown in #file:LEARNING_PROCESS_EXPLAINED.md   . Lats  do a step by step data analyse following the official documentation of what data each part of the system is expecting and what its outputs shoul be. Also check if the debug system is correctly implemented, so we are 100% sure about our conclusions and analysis.. For each step and analyse or problem that we face fetch official documentation for context.

Lats continue with my question: (If our CNN is learning together with out TD3 agent, where the CNN learning is saved so we can use it later on in the deploy part, the same with the TD3 agent, where its learning is saved so we use it later on in deploy?)  of our data pipeline, fetch relevant documentation about this step so we know what data it gets and what data its suppose to output. Also read relevant papers in #contextual about this step we are about to analyse.
We need to analyse using official documentation if our system is getting the correct data, correctly using it and transforming and then correctly outputing it.

Fetch additional info about the problem in TD3 DRL paper in contextual folder and in CARLA and Gymnasium Documentation.


Before each bug solution  you MUST fetch documentation information about the context of the specific problem we are  about to solve in order to find buges and improvements.

Fetch Carla documentation, and read #contextual folder related papers for better analyze context and the #TD3 #file:TD3.py #file:utils.py  folder contains the orignal TD3 DRL proposed in the #file:Addressing Function Approximation Error in Actor-Critic Methods.tex  paper. Above all you must be 100% sure of your proposed conclusion/solution, and MUST back it up (validate) the conclusion/solution with official documentation for Carla 0.9.16 and TD3. The papers in contextual folder are a related works that uses TD3/CNN/DRL in CARLA, read it for context.

Our implementation should focus the simplicity in order to achieve our final paper #file:ourPaper.tex

Remember, you must ALWAYS fetch Latest documentation contextual information at docs in order to reference your analyses and codes implementation. Fetch docs related to the analyses we are about to do or the code we are about to implemented:

https://carla.readthedocs.io/en/latest/build_docker/

https://carla.org/2025/09/16/release-0.9.16/

https://carla.readthedocs.io/en/latest/python_api/

https://carla.readthedocs.io/en/latest/tutorials/

https://stable-baselines3.readthedocs.io/en/master/modules/td3.html#


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

