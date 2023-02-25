'''
API Docs Found Here:
https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Python-LLAPI-Documentation.md#mlagents_envs.base_env.DecisionStep
'''

import mlagents
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.base_env import (
    BaseEnv,
    DecisionSteps,
    TerminalSteps,
    BehaviorSpec,
    ActionTuple,
    BehaviorName,
    AgentId,
    BehaviorMapping,
)

env = UE(file_name='Pacman', seed=1, side_channels=[])

env.reset()

behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]

#print(spec) shows you all thep properties contained in spec you can access
#This prints the shape of the numpy array of observationSpecs
print(spec.observation_specs[0].shape)

#spec.action_spec gets access to the ActionSpec - a tuple containing info on type of agent action and other action info
print(spec.action_spec)
agentActionSpec = spec.action_spec

if agentActionSpec.is_continuous():
    print("Action is continuous")

if agentActionSpec.is_discrete():
    print("Action is discrete")

#Can use ActionSpec to generate random action by calling agentActionSpec.random_action(#)
#Where # is the number of agents
#Print discrete array or access by randAction.discrete
randAction = agentActionSpec.random_action(1)

env.reset()

#Gets DecisionSteps object and TerminalSteps object
decision_steps, terminal_steps = env.get_steps(behavior_name)

#list(decision_steps) gives you the id of the agent who is requesting a decision

# Shows how you access the observations in decision steps
print(decision_steps.obs)

#Setting some loop params
trackedAgent = -1
done = False
episodeRewards = 0
#RHS returns the id of the agent
trackedAgent = decision_steps.agent_id[0]
#decision_steps.__getitem__(0) where 0 can be replaced by agentid gives you decision step for specific agent

while not done:
    if trackedAgent == -1 and len(decision_steps) >= 1:
        trackedAgent = decision_steps.agent_id[0]

    randAction = agentActionSpec.random_action(1)

    env.set_actions(behavior_name, randAction)

    env.step()

    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if trackedAgent in decision_steps:
        episodeRewards += decision_steps[trackedAgent].reward

    if trackedAgent in terminal_steps:
        episodeRewards += terminal_steps[trackedAgent].reward
        done = True

print("Total rewards for episode is " + str(episodeRewards))    

#Creating a custom ActionTuple
#For our case we have ActionTuple.continuous of shape (1,0) and discrete of shape (1,1)
#So need to do this:
#randAction = ActionTuple(np.zeros((1,0)), np.array([[1]]))
#second entry of ActionTuple can be whatever number the NN outputs e.g.