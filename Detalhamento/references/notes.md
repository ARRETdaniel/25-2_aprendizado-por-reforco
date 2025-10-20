Take the role of an experienced RL researcher, you must **challenge assumptions**, **test logic**, propose **alternative perspectives**, and **prioritize safety, correctness, reproducibility, and reusability** over convenience. You must write clean, professional, reusable code/text and easy to maintenance. Following best practice for code/text writing: Concise, coherent. Follow the Latest ABNT 2 (2025) norm. Write in an academic manner.


TD3 documentation https://spinningup.openai.com/en/latest/algorithms/td3.html#


1. [Towards Robust Decision-Making for Autonomous Highway Driving Based on Safe Reinforcement Learning](file:///C:/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/ref/2024%20Towards%20Robust%20Decision-Making%20for%20Autonomous%20Highway%20Driving%20Based%20on%20Safe%20Reinforcement%20Learning.pdf)
Page 19, shows  a experimental Setup that may be useful for our project.

2. [Implementing Deep Reinforcement Learning in Autonomous Control Systems](file:///C:/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/ref/2024%20Implementing%20Deep%20Reinforcement%20Learning%20in%20Autonomous%20Control%20Systems.pdf)
Has a excellent introduction comparing RL algorithms with other approaches.  (Copy it as contentxt for our introduction)

3. [Deep Reinforcement Learning for Autonomous Vehicle Intersection Navigation - CARLA](file:///C:/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/ref/2023%20Deep%20Reinforcement%20Learning%20for%20Autonomous%20Vehicle%20Intersection%20Navigation%20-%20CARLA.pdf)
 Twin Delayed DDPG [19] has emerged as a relevant
 and promising algorithm for autonomous vehicles due to
 its stability, reduced overestimation bias, and improved
 exploration capabilities. TD3 is an off-policy actor-critic
 algorithm that extends DDPG by incorporating three key
 enhancements:
 1) Twin Q-networks: which are used to mitigate
 overestimation bias by maintaining two separate Q-function
 approximators and taking the minimum value of the two. Similar to the double Q-learning*.
 2) Delayed policy updates: wherein the actor and target
 networks are updated less frequently than the Q-networks to
 improve stability.
 3) Target policy smoothing: which adds noise to the target
 actions during the learning process to encourage exploration
 and prevent overfitting to deterministic policies.
 These improvements have enabled TD3 to achieve superior
 performance in a variety of tasks, including intersection
 management, compared to its predecessor DDPG [19].
 Additionally, TD3 offers an efficient approach for learning
 complex decision-making policies required for navigating
 intersections safely, making it well-suited for autonomous
 driving applications

4. [Deep reinforcement learning based control for Autonomous
Vehicles in CARLA](file:///C:/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/ref/2022%20Deep%20reinforcement%20learning%20based%20control%20for%20Autonomous%20Vehicles%20in%20CARLA.pdf)

 2.3  Deep reinforcement learning
 While Reinforcement learning (RL) algorithms are dynamically learning with a trial
and error method to maximize the outcome, being rewarded for a correct prediction and
penalized for incorrect predictions, and successfully tested for solving Markov Decision
Problems (MDPs). However, as illustrated above, it can be overwhelming for the algo
rithm to learn from all states and determine the reward path. Then, DRL based algo
rithms replaces tabular methods of estimating state values (need to store all possible
state and value pairs) with a function approximation (the Deep prefix comes here) that
enables the agent, in this case the ego-vehicle, to generalize the value of states it has
never seen before, or has partial by seen, by using the values of similar states. Regarding
this, the combination of Deep Learning techniques and Reinforcement Learning algo
rithms have demonstrated its potential solving some of the most challenging tasks of
autonomous driving, such as decision making and planning [49]. Deep Reinforcement
Learning (DRL) algorithms include: Deep Q-learning Network (DQN) [17, 33], Double-
DQN, actor-critic (A2C, A3C) [27], Deep Deterministic Policy Gradient (DDPG)
[45, 47] and Twin Delayed DDPG (TD3) [50]. Our work is focused in DQN and DDPG
algorithms, which are explained in the following section. (My work will focus on TD3 DDPG)



1. [An Empirical Study of DDPG and PPO-Based Reinforcement Learning Algorithms for Autonomous Driving](file:///C:/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/ref/2023%20An%20Empirical%20Study%20of%20DDPG%20and%20PPO-Based%20Reinforcement%20Learning%20Algorithms%20for%20Autonomous%20Driving.pdf)
Has a nice table for TABLE 2. Summary of the related literature.
