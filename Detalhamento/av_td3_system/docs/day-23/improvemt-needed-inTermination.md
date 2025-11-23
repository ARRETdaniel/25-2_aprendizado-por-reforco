Evidence B (Lane Invasions): Avg Lane Invasions: 1.00. Every single episode ends with a lane invasion. This implies your environment is set to terminate immediately when a line is crossed.


The "Bug": The paper End-to-End Deep Reinforcement Learning for Lane Keeping Assist explicitly states: "We concluded that the more we put termination conditions, the slower convergence time to learn". By killing the episode immediately upon touching a line, you prevent the agent from learning how to recover from a mistake.

Immediate Fixes
To validate if the agent can learn, apply these changes based on the papers:

Disable Termination on Lane Invasion: Only terminate the episode on a Collision or if the car is completely off-road (e.g., > 2 meters from center). Allow the agent to cross the line, take the penalty, and try to steer back.
