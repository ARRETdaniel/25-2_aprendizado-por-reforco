Comprehensive Diagnostic Report on TD3 Agent Convergence Failure in CARLA Autonomous Navigation
1. Introduction: The Dynamics of Learning Instability in Autonomous Driving
The deployment of Deep Reinforcement Learning (DRL) agents within high-fidelity simulation environments such as CARLA represents one of the most challenging frontiers in modern robotics research. The user’s specific scenario—a Twin Delayed Deep Deterministic Policy Gradient (TD3) agent tasked with point-to-point navigation in CARLA Town 01, utilizing a multimodal architecture of Convolutional Neural Networks (CNN) for visual perception and kinematic vectors for state estimation—presents a complex optimization landscape. The reported failure mode, characterized by immediate divergence into "hard right/left" steering or "staying still" behaviors precisely upon the commencement of the learning phase (post-10,000 steps of exploration), is a paradigmatic example of the fragility inherent in continuous control DRL.1
This behavior is not merely a symptom of suboptimal hyperparameters but rather a manifestation of fundamental conflicts between the initialization of deep neural networks, the topology of the action space, and the sparsity of the reward signal in autonomous driving tasks. The transition from a purely exploratory policy (random noise) to a learned policy (gradient-driven) at the 10,000-step mark creates a "shock" to the system. If the replay buffer is populated primarily with failure cases—which is statistically probable in random exploration within a constrained urban environment—the critic networks will approximate a value landscape where "existence" is correlated with "negative reward." Consequently, the actor network, driven to maximize this value, converges rapidly to the boundaries of the action space (saturation) or to a zero-energy state (inaction) to mitigate the accumulation of penalties.3
This report provides an exhaustive, dissertation-level analysis of these failure modes. It dissects the mathematical underpinnings of TD3’s actor-critic interaction under the specific constraints of the CARLA simulator, explores the perils of multimodal feature fusion where scalar kinematic data can overwhelm high-dimensional visual features, and proposes a rigorous remediation strategy grounded in the latest literature on "Learning by Cheating," warm-start protocols, and dense reward shaping.5 The analysis moves beyond surface-level debugging to investigate the second- and third-order effects of sensor fusion and reward dynamics that drive the agent into these specific local optima.
2. Theoretical Framework: TD3 and the Cold Start Problem in CARLA
To understand why the agent exhibits catastrophic behavior immediately after the exploration phase, one must first deconstruct the mechanics of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm and how it interacts with the specific physics and penalties of the CARLA environment.
2.1 The Mechanics of TD3 in Continuous Control
TD3 is an off-policy algorithm designed to address the overestimation bias of DDPG. It employs two critic networks, $Q_{\theta_1}$ and $Q_{\theta_2}$, and uses the minimum of their outputs to form the target for the Bellman update.


$$y = r + \gamma \min_{i=1,2} Q_{\theta'_i}(s', \pi_{\phi'}(s') + \epsilon)$$

This mechanism is generally stabilizing. However, in the context of the user's "cold start" scenario, it can contribute to the "staying still" pathology. During the first 10,000 steps, the agent collects data using a random policy. In CARLA Town 01, a random policy results in a high frequency of collisions (negative rewards) and very few successful lane-following sequences.
When learning starts, the critics $Q_1$ and $Q_2$ are updated to reflect this history of failure. Because the minimum operator is used, the target value $y$ is pessimistic. If the agent acts (moves), it crashes. Therefore, the expected return for any action with a non-zero velocity vector is highly negative. The actor network $\pi_\phi$, updated to maximize $Q_{\theta_1}$, seeks regions of the state-action space with the least negative value. If the reward function does not strictly penalize inaction (or penalizes it less than a collision), the "safe" local minimum is to output zero throttle and zero steering.7 The "Twin" mechanism, designed to prevent optimistic overestimation of risky actions, essentially works too well: it confirms with high confidence that moving is dangerous.
2.2 The "Shock" of the First Gradient Update
The user notes that the bad behavior begins "when the learning phase starts." This transition point is critical. At step 10,000, the network shifts from:
Data Collection Mode: Actions are determined by $a_t = \text{clip}(\mu(s_t) + \mathcal{N}, -1, 1)$ or purely uniform sampling.
Training Mode: The weights of the neural networks are updated via backpropagation.
If the learning rate is standard (e.g., $10^{-3}$ for the Actor) and the batch size is standard (e.g., 100 or 256), the first gradient update is calculated over a batch of transitions that are likely 90% collisions.8 The loss function for the critic is the Mean Squared Error (MSE) of the Bellman residual. Given the high magnitude of collision penalties (often -100 or more), the gradients generated in this first step can be enormous.
When these gradients are backpropagated to the Actor, they can cause a "weight explosion." The weights of the final layer, originally initialized to small values, may jump to large magnitudes in a single update. As a result, the pre-activation values for the tanh function become very large (e.g., $>3.0$ or $<-3.0$). This forces the tanh output to saturate at $+1.0$ or $-1.0$. This explains the "hard right" or "hard left" behavior: the network has been shocked into a saturated state from which it cannot recover because the gradient of a saturated tanh is near zero.10
2.3 The Replay Buffer Distribution at Step 10,000
The contents of the replay buffer at the moment learning starts are the primary determinant of initial convergence. In the user’s implementation, the buffer is filled with 10,000 steps of exploration.
Town 01 Characteristics: Town 01 consists of simple roads but features curbs and sidewalks. A random agent will drive off the road or into a wall within seconds.
Data Imbalance: A typical episode length for a random agent might be 50-100 steps. This means the buffer contains roughly 100-200 "crash" events. The successful navigation samples are practically non-existent.
Implication: The Critic learns a value surface that is effectively flat and negative everywhere except for the exact states of collision, where it is deeply negative. The Actor has no gradient pointing towards "good" driving because "good" driving is not represented in the buffer. It only has gradients pointing away from crashes. The vector summation of "away from left wall" and "away from right wall" often results in "stay still" or, due to noise, a commitment to one extreme.11
3. Diagnostic Analysis: The "Hard Turn" Anomaly
The "hard right" or "hard left" behavior is the most distinct failure mode reported. It suggests a mechanical failure in the action generation process, specifically related to the properties of the activation function and the scaling of the action space.
3.1 The Geometry of Tanh Saturation
The TD3 actor network universally ends with a Hyperbolic Tangent (tanh) activation function to bound actions to the range $[-1, 1]$.


$$\pi(s) = \tanh(W_L h_{L-1} + b_L)$$

The derivative of tanh is:


$$\frac{\partial \pi}{\partial z} = 1 - \tanh^2(z)$$

This derivative describes the "learnability" of the network. When the output is near 0 (linear region), the derivative is near 1, allowing gradients to flow backward efficiently. However, as the absolute value of the pre-activation $z$ increases, the derivative approaches zero.
If the initialization of the network weights $W_L$ is too large, or if the first gradient update is too aggressive, the value of $z$ can easily exceed $\pm 2.0$.
At $z = 2.0$, $\pi(s) \approx 0.96$.
At $z = 3.0$, $\pi(s) \approx 0.995$.
At $z = 10.0$ (common after a gradient explosion), $\pi(s) \approx 1.0$.
In this saturated state, the gradient is practically zero. The agent is mathematically incapable of learning to steer "less," because the feedback signal (that a hard right turn led to a crash) cannot pass through the saturated activation function to correct the weights.12 The agent becomes "locked" in a hard turn. This is consistent with the user's observation that the behavior starts immediately upon learning.
3.2 Action Space Mapping Mismatch
A critical and often overlooked aspect of RL in CARLA is the translation of the agent's dimensionless output $[-1, 1]$ to the simulator's physical control inputs. The carla.VehicleControl class expects:
steer: $[-1.0, 1.0]$
throttle: $[0.0, 1.0]$
brake: $[0.0, 1.0]$
If the user’s agent outputs a vector $a \in [-1, 1]^2$, a common naive mapping is:
steer = $a$
throttle = $a$
This mapping is catastrophic for the throttle. If $a$ is negative (which occurs 50% of the time with random initialization), the throttle is effectively 0 (or negative, which CARLA might interpret as ignoring or clipping). If the agent learns that $a = -1.0$ (brake) prevents crashes, it will output -1.0.
Furthermore, the steering range of $[-1.0, 1.0]$ in CARLA corresponds to the full lock of the steering wheel (e.g., 70 degrees). At any speed above 10 km/h, a steering input of 1.0 is physically violent. It causes immediate loss of traction and lateral instability.
Insight: The "Hard Turn" is not just a saturated network; it is a physically excessive action. A steering value of $0.5$ is usually a sharp turn. A value of $1.0$ is an emergency maneuver. If the agent initializes to random weights, it outputs values distributed across $[-1, 1]$. The simulator interprets these as extreme maneuvers, leading to immediate failure.14
3.3 Weight Initialization Protocols
To prevent this, the distribution of initial actions must be concentrated around 0. Standard initialization schemes (Xavier/Glorot) are designed to maintain variance through the network, but for the final layer of an RL actor, this variance is often too high.
Standard Initialization: Output variance $\approx 1.0$. Actions range $[-1, 1]$.
Required Initialization: Output variance $\approx 0.01$. Actions range $[-0.1, 0.1]$.
This is achieved by initializing the final layer's weights from a uniform distribution $U[-3 \times 10^{-3}, 3 \times 10^{-3}]$.15 This ensures the agent starts by driving straight. The user’s implementation likely lacks this specific constraint, leading to the immediate hard turn behavior.
4. Diagnostic Analysis: The "Staying Still" Anomaly
The second failure mode—the agent refusing to move—is a rational response to a pathological reward landscape.
4.1 The Local Minimum of Safety
In the absence of a "safe" driving policy, the safest action is inaction. The user's reward function gives "positive rewards along the way if it does actions that leads to goal." This implies a distance-based reward:


$$R_t = d_{t-1} - d_t$$

However, the user also notes negative rewards for "going over the side walk" and "lane invasion."
Magnitude Disparity: The reward for moving 1 meter toward the goal might be $+0.1$. The penalty for a lane invasion is often $-20$ or $-50$.
Risk Assessment: To get $+10$ total reward, the agent must drive successfully for 100 meters. One mistake destroys this accumulated reward.
Critic's View: During the first 10k steps, the agent tried to move and mostly failed. The Critic estimates $Q(s, \text{move}) \approx -50$. The Critic estimates $Q(s, \text{stop}) \approx 0$ (assuming no time penalty) or a small negative value (step penalty).
Convergence: Since $0 > -50$, the Actor converges to "Stop."
4.2 The "Do Nothing" Loop
This is a classic RL failure mode known as the "Do Nothing" loop. It occurs when the exploration capability of the agent is insufficient to find the sparse positive reward (reaching the goal) frequently enough to offset the dense negative rewards (collisions).
Evidence: The user mentions the agent "has to navigate for a long time in order to achieve goal." This confirms the reward is sparse/delayed.
Constraint: Without a dense "survival reward" or a "velocity reward" that is strictly positive for staying on the road, the agent sees no benefit in moving. If the user does have a velocity reward, but the collision penalty is too high, the expected return is still negative.3
5. Multimodal Sensor Fusion Architecture and Modality Collapse
The user's architecture fuses visual data (CNN) with kinematic vectors (Speed, Waypoints). This fusion is a notorious source of instability in End-to-End Autonomous Driving.
5.1 The Modality Collapse Phenomenon
When a neural network receives inputs from multiple modalities, it tends to overfit to the modality that is "easiest" to learn—i.e., the one with the strongest correlation to the reward signal and the simplest gradient path.
The Competitors:
Image (CNN): High-dimensional ($3 \times 256 \times 256$), highly non-linear, requires finding edges, lanes, and depth. Gradient path is deep and noisy.
Speed (MLP): Scalar value. Highly correlated with "velocity reward." Gradient path is shallow and direct.
Waypoints (MLP): Low-dimensional vectors. Correlated with "distance to goal."
The Collapse: The network quickly learns that "Higher Speed = Higher Reward" (initially). It ramps up the throttle. However, to steer correctly, it must look at the image. But learning to process the image takes thousands of epochs. Learning to press the throttle takes 10 epochs.
Result: The agent learns to throttle up (based on the speed vector) but fails to learn to steer (based on the image). The result is a fast car that goes straight into a wall. When it crashes, the negative reward makes the agent stop. It never connects the image pixels to the steering command because the gradient signal from the image is drowned out by the strong gradients from the speed/waypoint vectors.16
5.2 Feature Normalization Failure
Another likely cause of the "hard turn" is the lack of normalization in the vector inputs.
Scenario: The user feeds raw waypoints $(x, y)$ into the network. In CARLA, map coordinates can be in the range of hundreds or thousands (e.g., $x=300.0$).
Impact: If the image features are normalized (outputs of CNN are usually small), but the waypoint features are massive ($300.0$), the dense layers following the fusion point will be dominated by the waypoint values.
Gradients: The gradients with respect to the weights connected to the waypoints will be orders of magnitude larger than those connected to the CNN. The optimizer will essentially ignore the CNN. The agent becomes blind, steering based on GPS coordinates alone, which is insufficient for obstacle avoidance or lane keeping.18
6. Comprehensive Remediation Strategy
To resolve these issues, a multi-layered approach involving architectural changes, initialization protocols, and reward shaping is required.
6.1 Phase 1: Fixing the Action Space and Initialization
The immediate fix for the "hard turn" is to constrain the actor's output.
Initialization: Explicitly initialize the weights of the final layer of the actor network to a uniform distribution with a very small range.
Protocol: $w \sim U[-3 \times 10^{-3}, 3 \times 10^{-3}]$, $b = 0$.
Effect: This ensures that at step 0 of learning, the pre-activation $z \approx 0$, so $\tanh(z) \approx 0$. The agent starts by coasting, not turning.
Action Scaling: Implement a robust wrapper for the CARLA VehicleControl.
Steering: Scale the network output. If the network outputs $a_{steer} \in [-1, 1]$, the applied steering should be $0.5 \times a_{steer}$. This limits the physical wheel angle to $\approx 35^\circ$, preventing instantaneous loss of control.14
Throttle/Brake: Use separate network outputs or a split logic, but ensure the mapping is smooth. A reliable mapping is:
Network output $a \in [-1, 1]$.
Throttle $= \max(0, a)$.
Brake $= \max(0, -a)$ (or simply 0 during early training to encourage movement).
6.2 Phase 2: Resolving the "Cold Start" with Autopilot
To solve the "Staying Still" problem, the replay buffer must contain examples of successful driving before learning starts.
Method: Instead of 10k steps of random exploration, perform 10k steps of Autopilot exploration.
Implementation: Use CARLA's built-in vehicle.set_autopilot(True). Record the observations and actions taken by the autopilot into the replay buffer.
Why: This fills the buffer with transitions where (State $\rightarrow$ Action $\rightarrow$ High Positive Reward).
Result: When learning starts, the Critic learns that specific actions (following the lane) yield high value. The Actor will then be updated to mimic these high-value actions. This is often referred to as "Behavioral Cloning warm-start" or "Learning from Demonstrations".19
6.3 Phase 3: Mitigating Modality Collapse
To ensure the agent uses the CNN, the feature fusion must be balanced.
Normalization: Ensure all vector inputs (speed, waypoints) are normalized to the range $[-1, 1]$ or $$.
Speed: $v_{norm} = v_{current} / v_{max}$ (e.g., $30/50$).
Waypoints: Use relative coordinates (ego-centric), not global. Divide by the maximum detection distance.
Late Fusion with LayerNorm: Apply Layer Normalization to the concatenated feature vector (CNN features + Vector features) before passing it to the decision layers. This standardizes the variance across modalities, preventing the speed vector from dominating the gradient.18
6.4 Phase 4: Dense Reward Shaping
The reward function must be reshaped to guide the agent out of the "stop" local minimum.
Proposed Reward Function:


$$R = w_v (v_{lon}) - w_c (|d_{center}|) - w_s (|a_{steer}|) + w_d (d_{goal}) + p_{survival}$$
$v_{lon}$ (Longitudinal Velocity): Strictly positive reward for moving forward.
$d_{center}$ (Lane Centering): Penalty for deviating from the center (dense signal).
$a_{steer}$ (Steering Penalty): Small penalty for high steering angles to encourage smooth driving.
$p_{survival}$: A small positive reward for every step the agent does not crash.
Crucial Addition: If $v < 1.0$ m/s, add a Stopping Penalty (e.g., -1.0). This makes "staying still" strictly worse than moving slowly.21
7. Comparison with Related Work and Literature Review
The challenges encountered by the user are well-documented in the autonomous driving RL literature.
7.1 TD3 vs. PPO in CARLA
While the user is employing TD3, snippet and 2 highlight that many successful implementations utilize PPO (Proximal Policy Optimization). PPO's "Trust Region" constraint prevents the policy from changing too drastically in a single update. This offers natural protection against the "Hard Turn" issue, as the agent cannot jump from "random" to "saturated" in one step. However, TD3 is theoretically more sample-efficient. The failure of TD3 here is likely due to the lack of entropy regularization (found in SAC) which would prevent the policy from collapsing to a single deterministic "stop" action.22
7.2 The "Learning by Cheating" Paradigm
A dominant theme in CARLA research, as noted in snippets 5, is the "Learning by Cheating" (LBC) approach. This method argues that training a vision-based agent from scratch (End-to-End RL) is inefficient. LBC proposes training a "privileged" agent first (using ground-truth map data) and then distilling this policy into a sensorimotor (vision) agent. The user's struggle with the vision-based TD3 agent validates the LBC hypothesis: without a teacher or privileged information, the correlation between pixels and steering is too difficult to learn before the agent gives up and stops.
7.3 Sensor Fusion Best Practices
Snippet 16 emphasizes that "multimodal fusion techniques... require systematic investigation." The simple concatenation of CNN and MLP outputs is often insufficient. State-of-the-art approaches (TransFuser) utilize Transformer-based attention mechanisms to fuse the LiDAR/Image features with the kinematic goals, allowing the network to dynamically attend to the visual features relevant to the current waypoint. While implementing a Transformer might be out of scope for the user's current debugging, it highlights that the fusion mechanism is a primary suspect for the "blind" driving behavior.
8. Implementation Tables and Data
To assist the user in restructuring their implementation, the following tables summarize the critical parameters and configurations derived from the analysis.
Table 1: Comparative Analysis of Initialization Protocols
Parameter
Standard (Current)
Recommended (Remediation)
Impact on Convergence
Actor Final Layer Weights
Xavier / He Normal
Uniform $[-3e^{-3}, 3e^{-3}]$
Prevents initial tanh saturation and hard turns.
Actor Final Layer Bias
0.0 or Random
0.0
Ensures centered distribution.
Exploration Noise (Training)
$\mathcal{N}(0, 0.1)$
$\mathcal{N}(0, 0.1)$ (Decaying)
Standard noise is fine if actions are centered.
Replay Buffer Content
10k Random Steps
10k Autopilot Steps
Provides positive examples for the Critic to learn from.
Actor Learning Rate
$10^{-3}$
$10^{-4}$
Slows down policy shifts, preventing oscillation.

Table 2: Action Space Mapping Strategy
Network Output (a)
Variable
Formula
Physical Meaning
$a \in [-1, 1]$
Steer
$S = 0.5 \times a$
Limits wheel angle to safe range ($\approx 35^\circ$).
$a \in [-1, 1]$
Throttle
$T = \max(0, a)$
Maps positive output to throttle.
$a \in [-1, 1]$
Brake
$B = \max(0, -a)$
Maps negative output to brake.

Table 3: Reward Function Composition for Robust Learning
Component
Weight (w)
Description
Purpose
Velocity ($v_{lon}$)
$+1.0$
Proportional to speed (m/s)
Incentivizes movement.
Centering ($d_{center}$)
$-0.5$
Distance from lane center (m)
Incentivizes precision.
Steering ($a_{steer}$)
$-0.1$
Steering magnitude
Incentivizes smoothness.
Collision
$-100.0$
Binary event
Termination penalty.
Lane Invasion
$-20.0$
Binary event
Soft penalty (non-terminal).
Standstill
$-1.0$
If $v < 0.1$ m/s
Critical: Prevents "Staying Still" local optimum.

9. Conclusion
The failure of the user's TD3 agent in CARLA Town 01 is not a result of a single coding error but a convergence of three distinct pathological factors common in sensorimotor reinforcement learning: Action Space Saturation, Multimodal Gradient Imbalance, and Reward-Induced Local Optima.
The "hard right/left" behavior is a mechanical consequence of unconstrained network initialization interacting with a sensitive action wrapper. The "staying still" behavior is a rational optimization strategy by the agent to minimize risk in the face of a sparse reward landscape and a history of failures (the random exploration phase). The "failure to learn" is exacerbated by the modality collapse where strong gradients from kinematic vectors blind the agent to visual inputs.
By implementing the remediation strategy outlined above—specifically the uniform small-weight initialization, autopilot warm-start, and dense reward shaping with standstill penalties—the user can destabilize the "stop" local minimum and provide the agent with the gradient slope necessary to discover valid driving policies. This transition from "Cold Start" to "Warm Start" is the decisive factor in successfully training deep reinforcement learning agents in complex autonomous driving simulators.
10. Detailed Code-Level Recommendations (Pseudocode)
10.1 Correct Network Initialization (PyTorch)

Python


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Apply specifically to the actor's output layer
# Assuming actor.l3 is the final layer
actor.l3.weight.data.uniform_(-3e-3, 3e-3)
actor.l3.bias.data.uniform_(-3e-3, 3e-3)


10.2 Multimodal Feature Fusion Block

Python


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # CNN for Image
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        #... more conv layers...
        self.fc_img = nn.Linear(conv_out_size, 256)

        # MLP for Vectors (Speed, Waypoints)
        self.fc_vec = nn.Linear(vector_input_dim, 256)

        # Fusion with Layer Norm
        self.ln = nn.LayerNorm(512) # 256 + 256
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, img, vec):
        x_img = F.relu(self.fc_img(self.cnn(img)))
        x_vec = F.relu(self.fc_vec(vec))

        # Concatenate
        x = torch.cat([x_img, x_vec], dim=1)

        # Apply Layer Norm to balance gradients
        x = self.ln(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


10.3 Autopilot Data Collection Loop

Python


# During the "Fill Buffer" phase (First 10k steps)
if total_steps < 10000:
    # Engage Autopilot
    vehicle.set_autopilot(True)

    # Retrieve the action the autopilot IS taking
    control = vehicle.get_control()
    action = [control.steer, control.throttle - control.brake]

    # Store this valid tuple in buffer
    replay_buffer.add(state, action, next_state, reward, done)

else:
    # Switch to Policy
    vehicle.set_autopilot(False)
    action = agent.select_action(state)


This comprehensive approach addresses the user's problem from the theoretical root to the practical implementation, ensuring a robust solution to the convergence failure.
Works cited
A Comprehensive Review of Reinforcement Learning for Autonomous Driving in the CARLA Simulator - arXiv, accessed December 2, 2025, https://arxiv.org/html/2509.08221v1
Think2Drive: Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-v2) - arXiv, accessed December 2, 2025, https://arxiv.org/html/2402.16720v2
TD3 agent fails to explore again after hitting the max action and gets stuck at the max action value. Additionally, the Q0 va... - MATLAB Answers - MathWorks, accessed December 2, 2025, https://www.mathworks.com/matlabcentral/answers/2129201-td3-agent-fails-to-explore-again-after-hitting-the-max-action-and-gets-stuck-at-the-max-action-value
Twin Delayed DDPG — Spinning Up documentation - OpenAI, accessed December 2, 2025, https://spinningup.openai.com/en/latest/algorithms/td3.html
Learning by Cheating - Vladlen Koltun, accessed December 2, 2025, http://vladlen.info/papers/learning-by-cheating.pdf
[1912.12294] Learning by Cheating - arXiv, accessed December 2, 2025, https://arxiv.org/abs/1912.12294
TD3: Overcoming Overestimation in Deep Reinforcement Learning | by Dong-Keon Kim, accessed December 2, 2025, https://medium.com/@kdk199604/td3-overcoming-overestimation-in-deep-reinforcement-learning-c52d1cc9d69a
New TD3 hyperparameters really improve the performance? · Issue #21 - GitHub, accessed December 2, 2025, https://github.com/sfujim/TD3/issues/21
[D] Batch size vs learning rate : r/MachineLearning - Reddit, accessed December 2, 2025, https://www.reddit.com/r/MachineLearning/comments/1fqqfos/d_batch_size_vs_learning_rate/
What do you mean by saturation in neural network training? Discuss the problems associated with saturation - AIML.com, accessed December 2, 2025, https://aiml.com/what-do-you-mean-by-saturation-in-the-context-of-neural-network-training-discuss-the-problems-associated-with-saturation/
Why is 100% exploration bad during the learning stage in reinforcement learning?, accessed December 2, 2025, https://ai.stackexchange.com/questions/22235/why-is-100-exploration-bad-during-the-learning-stage-in-reinforcement-learning
Tanh Activation in Neural Network - GeeksforGeeks, accessed December 2, 2025, https://www.geeksforgeeks.org/deep-learning/tanh-activation-in-neural-network/
Neural Network with tanh wrong saturation with normalized data - Stack Overflow, accessed December 2, 2025, https://stackoverflow.com/questions/13632976/neural-network-with-tanh-wrong-saturation-with-normalized-data
Sending actions to the CARLA simulation server - Hands-On Intelligent Agents with OpenAI Gym [Book] - O'Reilly, accessed December 2, 2025, https://www.oreilly.com/library/view/hands-on-intelligent-agents/9781788836579/aaa23b6d-3f09-4afe-8b56-f01b73a60685.xhtml
Introduction to Reinforcement Learning (DDPG and TD3) for News Recommendation, accessed December 2, 2025, https://medium.com/data-science/reinforcement-learning-ddpg-and-td3-for-news-recommendation-d3cddec26011
Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines - PMC, accessed December 2, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC7567861/
Deep Reinforcement Learning: 0 to 100 | Towards Data Science, accessed December 2, 2025, https://towardsdatascience.com/deep-reinforcement-learning-for-dummies/
Normalization Enhances Generalization in Visual Reinforcement Learning - IFAAMAS, accessed December 2, 2025, https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1137.pdf
Twin-Delayed Deep Deterministic (TD3) Policy Gradient Agent - MATLAB & Simulink, accessed December 2, 2025, https://www.mathworks.com/help/reinforcement-learning/ug/td3-agents.html
benwex93/WarmStartRL: Analyzing and Overcoming Degradation in Warm Start Off-Policy RL - GitHub, accessed December 2, 2025, https://github.com/benwex93/WarmStartRL
A Comparative Study of Deep Reinforcement Learning Algorithms for Urban Autonomous Driving: Addressing the Geographic and Regulatory Challenges in CARLA - MDPI, accessed December 2, 2025, https://www.mdpi.com/2076-3417/15/12/6838
Solving the Exploration-Exploitation Dilemma in Reinforcement Learning - Medium, accessed December 2, 2025, https://medium.com/data-scientists-diary/solving-the-exploration-exploitation-dilemma-in-reinforcement-learning-07b4c21e3d40
dotchen/LearningByCheating: (CoRL 2019) Driving in CARLA using waypoint prediction and two-stage imitation learning - GitHub, accessed December 2, 2025, https://github.com/dotchen/LearningByCheating
