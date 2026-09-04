"""RL fine-tuning of the RAP planner against the NAVSIM PDM score.

The original RAP pipeline is supervised: a frozen DINOv3 backbone feeds a
trajectory head, and the PDM score is used only as a training *signal* for the
scorer head. This package reuses exactly the same cached data, but treats the
PDM scorer as an *environment*: the policy proposes a trajectory, the scorer
returns a reward, collisions are punished.

Layout::

    rl/config.py       paths and hyper-parameters (one dataclass)
    rl/reward.py       PDM sub-scores -> scalar reward
    rl/residual.py     action -> trajectory; shared by the env and the agent
    rl/precompute.py   one offline pass: RAP cache -> compact RL observations
    rl/env.py          RAPPlanningEnv, a gymnasium.Env
    rl/train.py        stable-baselines3 PPO entry point
    rl/eval.py         held-out evaluation + reward-sensitivity probe
    rl/agent.py        RLResidualAgent: the policy as a NAVSIM agent, for the benchmark
    rl/parity_check.py asserts agent.py sees what precompute.py wrote

Note that rl/agent.py is NOT imported here. It pulls in navsim's agent stack and
stable-baselines3, and it is loaded by hydra inside the devkit's evaluation process;
importing it eagerly would make `import rl` expensive everywhere else.

See rl/README.md for the design rationale and the run order.
"""

from rl.config import RLConfig

__all__ = ["RLConfig"]
