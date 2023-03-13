from NeuralNetworkPackage.inputLayer import InputLayer
from NeuralNetworkPackage.convolutionLayer import ConvolutionalLayer
from NeuralNetworkPackage.flatteningLayer import FlatteningLayer
from NeuralNetworkPackage.poolingLayer import PoolingLayer
from NeuralNetworkPackage.maxPoolingCalc import MaxPoolingCalc
from NeuralNetworkPackage.fullyConnectedLayer import FullyConnectedLayer
from NeuralNetworkPackage.softmaxActivationLayer import SoftmaxActivationLayer
from NeuralNetworkPackage.crossEntropyLayer import CrossEntropyLayer
from NeuralNetworkPackage.model import Model
from collections import deque
import random
import mlagents
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
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

import numpy as np

#Constant board size params
NUM_MAP_ROWS = 12
NUM_MAP_COLS = 14

#Number of agents in game
NUM_AGENTS = 1

'''Training Params'''
#Max amount of exploration is 100%
MAX_EPSILON = 1
#Min amount of exploration by agent set to 1%
MIN_EPSILON = 0.01
#Epsilon greedy training strategy - begin at one (completely random)
EPSILON = 1
#Amount to scale epsilon decay by
DECAY = 0.01
#Number of training episodes
NUM_TRAINING_EPISODES = 300
NUM_TESTING_EPISODES = 100

replayMemory = deque(maxlen=50_000)

#Note - 0,1,2,3 action indices correspond to up, down, right, left respectively

def train(inReplayMemory, inTrainingModel, inTargetModel, inDone):
    learningRate = 0.7
    discountFactor = 0.618

    MIN_REPLAY_SIZE = 1000
    if (len(inReplayMemory) < MIN_REPLAY_SIZE):
        return
    
    batchSize = 8
    miniBatch = random.sample(inReplayMemory, batchSize)
    #Get the observations from minibatch
    currentStates = np.zeros((len(miniBatch), miniBatch[0][0].shape[1], miniBatch[0][0].shape[2]))
    #Populate currentStates with observations
    for idx, transition in enumerate(miniBatch):
        currentStates[idx] = transition[0]

    currentQsList = inTrainingModel.predict(currentStates)
    newCurrentStates = np.zeros((len(miniBatch), miniBatch[0][3].shape[1], miniBatch[0][3].shape[2]))
    #Populate newCurrentStates with subsequent observations
    for idx, transition in enumerate(miniBatch):
        newCurrentStates[idx] = transition[3]

    futureQsList = inTargetModel.predict(newCurrentStates)

    X = []
    Y = []
    for idx, (observation, action, reward, newObservation, done) in enumerate(miniBatch):
        if not done:
            maxFutureQ = reward + discountFactor * np.max(futureQsList[idx])
        else:
            maxFutureQ = reward

        currentQs = currentQsList[idx]
        actionNumerical = action.discrete[0][0]
        currentQs[actionNumerical] = (1 - learningRate) * currentQs[actionNumerical] + learningRate * maxFutureQ

        X.append(observation[0])
        Y.append(currentQs)

    inTrainingModel.train(np.array(X), np.array(Y))
    return

#Create empty random sized X array
X = np.array([np.random.randint(8, size=168).reshape((12,14))])

#Set up layers
il = InputLayer(X, False)
cl = ConvolutionalLayer(3)
mpc = MaxPoolingCalc()
pl = PoolingLayer(3, 3, 2, mpc)
fl = FlatteningLayer()
fcl = FullyConnectedLayer(20,4)
sal = SoftmaxActivationLayer()
cel = CrossEntropyLayer()

layers = [il, cl, pl, fl, fcl, sal, cel]

#Training model used for training and current q
trainingModel = Model(layers, 1)
#Target model used for future q and final model
targetModel = Model(layers, 1)

#Make a prediction
prediction = trainingModel.predict(X)
maxValIdx = np.argmax(prediction)

#Set action with predicted action
predictedAction = ActionTuple(np.zeros((1,0)), np.array([[maxValIdx]]))

#Create channel to specify run speed
channel = EngineConfigurationChannel()

#Open pacman environment
env = UE(file_name='../MiniGameMap/Pacman', seed=1, side_channels=[channel])

channel.set_configuration_parameters(time_scale= 5.0)

env.reset()

#Get the name of the behavior we're using
behaviorName = list(env.behavior_specs)[0]
#Get the behavior spec which contains observation data
spec = env.behavior_specs[behaviorName]

#spec.action_spec is a ActionSpec tuple containing info 
#on type of agent action and other action info
agentActionSpec = spec.action_spec

#Set action to predicted action
env.set_actions(behaviorName, predictedAction)

env.reset()

#Gets DecisionSteps object and TerminalSteps object
decisionSteps, terminalSteps = env.get_steps(behaviorName)

#Get the map state representation and reorder to grid size for CNN
decisionStepsObs = decisionSteps[0].obs
observation = np.array([np.reshape(decisionStepsObs, (NUM_MAP_ROWS, NUM_MAP_COLS))])

#Setting some loop params
trackedAgent = -1
done = False
episodeRewards = 0
#RHS returns the id of the agent
trackedAgent = decisionSteps.agent_id[0]
stepsToUpdateTargetModel = 0

for episode in range(NUM_TRAINING_EPISODES):
    episodeRewards = 0
    trackedAgent = -1
    done = False
    while not done:
        stepsToUpdateTargetModel += 1

        if trackedAgent == -1 and len(decisionSteps) >= 1:
            trackedAgent = decisionSteps.agent_id[0]

        randActionProb = np.random.rand()

        #Explore with epsilon greedy strategy
        if (randActionProb <= EPSILON):
            action = agentActionSpec.random_action(NUM_AGENTS)
        else:
            #Get predicted action
            prediction = trainingModel.predict(observation)
            maxValIdx = np.argmax(prediction)
            action = ActionTuple(np.zeros((1,0)), np.array([[maxValIdx]]))
        
        #Set action
        env.set_actions(behaviorName, action)
        env.step()

        decisionSteps, terminalSteps = env.get_steps(behaviorName)
        lastStepReward = 0
        if trackedAgent in decisionSteps:
            lastStepReward = decisionSteps[trackedAgent].reward
            episodeRewards += lastStepReward

        if trackedAgent in terminalSteps:
            lastStepReward = terminalSteps[trackedAgent].reward
            episodeRewards += lastStepReward
            done = True

        if not done:
            decisionStepsObs = decisionSteps[trackedAgent].obs
            newObservation = np.array([np.reshape(decisionStepsObs, (NUM_MAP_ROWS, NUM_MAP_COLS))])
        else:
            newObservation = observation

        replayMemory.append([observation, action, lastStepReward, newObservation, done])

        observation = newObservation

        #Update Final Model using Bellman Equation
        if stepsToUpdateTargetModel %8 == 0 or done:
            train(replayMemory, trainingModel, targetModel, done)

    #Update epsilon so we're less likely to take random action
    EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY * episode)
    
    print("Total rewards for episode is " + str(episodeRewards))    