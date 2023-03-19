from NeuralNetworkPackage.inputLayer import InputLayer
from NeuralNetworkPackage.convolutionLayer import ConvolutionalLayer
from NeuralNetworkPackage.flatteningLayer import FlatteningLayer
from NeuralNetworkPackage.poolingLayer import PoolingLayer
from NeuralNetworkPackage.maxPoolingCalc import MaxPoolingCalc
from NeuralNetworkPackage.fullyConnectedLayer import FullyConnectedLayer
from NeuralNetworkPackage.reLuLayer import ReLuLayer
from NeuralNetworkPackage.logisticSigmoidLayer import LogisticSigmoidLayer
from NeuralNetworkPackage.softmaxActivationLayer import SoftmaxActivationLayer
from NeuralNetworkPackage.crossEntropyLayer import CrossEntropyLayer
from NeuralNetworkPackage.adamWeightUpdateCalc import AdamWeightUpdateCalc
from NeuralNetworkPackage.squaredErrorLayer import SquaredErrorLayer
from NeuralNetworkPackage.model import Model
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

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

#Whether to serialize weights with custom file path (since Rob's run directory is different than Ram's)
USE_CUSTOM_FILE_PATH = False

#Whether we should normalize rewards
NORMALIZE_REWARDS = True

#Should we use a living reward
USE_LIVING_REWARD = False

#Living Reward
LIVING_REWARD = -1

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
#Amount to scale epsilon decay by after every episode
DECAY = 0.03
#Number of training episodes
NUM_TRAINING_EPISODES = 100
NUM_TESTING_EPISODES = 100
#Max number of stored replay memories
MAX_REPLAYMEMORY_SIZE = 1000
#Amount of Replay Memory to Delete When Clearing
AMT_REPLAY_MEM_TO_CLEAR_ON_CLEAR = 250
#Replay memory batch size for training
REPLAY_MEMORY_BATCH_SIZE = 250
#Minimum number of replays to train
MIN_REPLAY_SIZE = 500
#Number of steps till training training model
NUM_STEPS_TILL_TRAIN = 1250
#End episode after this many steps
NUM_STEPS_TILL_END_EPISODE = 2500
#Number of episodes until we train our target model
NUM_EPISODES_TILL_TRAIN_TARGET = 2

#assert NUM_STEPS_TILL_END_EPISODE <= NUM_REPLAY_MEMORY_TO_DELETE_AT_MAX, "Make sure you are setting num steps till training to be <= than replay batch size to delete"
assert REPLAY_MEMORY_BATCH_SIZE <= MIN_REPLAY_SIZE, "Min Replay Size must be at least as big as replay batch size"

#replayMemory = deque(maxlen=MAX_REPLAYMEMORY_SIZE)
replayMemory = np.zeros((MAX_REPLAYMEMORY_SIZE), dtype=np.ndarray)
toAddReplayMemIdx = 0
numAddedReplayMems = 0

#Note - 0,1,2,3 action indices correspond to up, down, right, left respectively

def train(inReplayMemory, inTrainingModel, inTargetModel, inNumAddedReplayMems, isDone):
    learningRate = 0.7
    discountFactor = 0.618

    # check to see if the training data has enough data points
    numAddedReplayMems = inNumAddedReplayMems
    if (numAddedReplayMems < MIN_REPLAY_SIZE):
        return numAddedReplayMems

    # shuffle the training data
    miniBatch = inReplayMemory[:numAddedReplayMems]
    np.random.shuffle(miniBatch)
    inReplayMemory = miniBatch
    miniBatch = miniBatch[:REPLAY_MEMORY_BATCH_SIZE]

    #Get the observations from minibatch
    currentStates = np.zeros((len(miniBatch), miniBatch[0][0].shape[1], miniBatch[0][0].shape[2]))

    #Populate currentStates with observations
    for idx, transition in enumerate(miniBatch):
        currentStates[idx] = transition[0]

    # produce y label for the final model
    currentQsList = inTrainingModel.predict(currentStates)

    #Populate newCurrentStates with subsequent observations
    newCurrentStates = np.zeros((len(miniBatch), miniBatch[0][3].shape[1], miniBatch[0][3].shape[2]))
    for idx, transition in enumerate(miniBatch):
        newCurrentStates[idx] = transition[3]

    # find best actions to take
    futureQsList = inTargetModel.predict(newCurrentStates)

    # assemble the x and y data sets
    obsNumRows = miniBatch[0][0].shape[1]
    obsNumCols = miniBatch[0][0].shape[2]
    numPredictions = currentQsList[0].shape[0]
    X = np.zeros((REPLAY_MEMORY_BATCH_SIZE, obsNumRows, obsNumCols))
    Y = np.zeros((REPLAY_MEMORY_BATCH_SIZE, numPredictions))
    for idx, (observation, action, reward, newObservation, done) in enumerate(miniBatch):
        if not done:
            qTarget = reward + discountFactor * np.max(futureQsList[idx])
        else:
            qTarget = reward

        currentQs = currentQsList[idx]
        actionNumerical = action.discrete[0][0]
        currentQs[actionNumerical] = (1 - learningRate) * currentQs[actionNumerical] + learningRate * qTarget

        X[idx] = observation[0]
        Y[idx] = currentQs

    # fit the training model
    inTrainingModel.train(X, Y)

    return numAddedReplayMems

def clearFirstXReplayMemories(inNumAddedReplayMems):
    remainingReplayMemSize = inNumAddedReplayMems - AMT_REPLAY_MEM_TO_CLEAR_ON_CLEAR
    replayMemory[:remainingReplayMemSize] = replayMemory[AMT_REPLAY_MEM_TO_CLEAR_ON_CLEAR:inNumAddedReplayMems]
    replayMemory[remainingReplayMemSize:inNumAddedReplayMems] = None
    inNumAddedReplayMems -= AMT_REPLAY_MEM_TO_CLEAR_ON_CLEAR 
    return inNumAddedReplayMems

def plot_metrics(log_root, log_file, model):
    log_df = pd.read_csv(f'{log_root}/{log_file}')

    # figure out what kind of loss function was used for the model
    if isinstance(model.layers[-1], SquaredErrorLayer):
        loss_str = 'Squared Error'
    elif isinstance(model.layers[-1], CrossEntropyLayer):
        loss_str = 'Cross Entropy'
    else:
        loss_str = ''
    
    # plot mean q value vs. episode Episode,Epsilon,Rewards,QValMean
    plt.figure(0)
    plt.plot(log_df.Episode, log_df.QValMean)
    plt.title('Mean Q vs. Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Q')
    plt.savefig(f'{log_root}/meanQ_vs_episode.png')

    # plot loss vs. epoch for each episode iteration on one graph
    plt.figure(1)
    epochs = np.arange(1, len(model.loss_arr)+1)
    plt.plot(epochs, model.loss_arr)
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epochs')
    plt.ylabel(f'Loss ({loss_str})')
    plt.savefig(f'{log_root}/loss_vs_epoch.png')

    # plot reward vs. episode
    plt.figure(2)
    plt.plot(log_df.Episode, log_df.Rewards)
    plt.title('Reward vs. Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.savefig(f'{log_root}/reward_vs_episode.png')

    # plot reward vs. epsilon
    plt.figure(3)
    plt.plot(log_df.Epsilon, log_df.Rewards)
    plt.title('Rewards vs. Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel(f'Rewards')
    plt.savefig(f'{log_root}/loss_vs_epsilon.png')

    # plot epsilon vs. episode
    plt.figure(4)
    plt.plot(log_df.Episode, log_df.Epsilon)
    plt.title('Epsilon vs. Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.savefig(f'{log_root}/epsilon_vs_episode.png')

    print(f"Plots saved here: {log_root}")

#Create empty random sized X array
X = np.array([np.random.randint(8, size=168).reshape((12,14)), np.random.randint(8, size=168).reshape((12,14))])

#Set up layers
tril = InputLayer(X, False)
tail = InputLayer(X, False)
trcl = ConvolutionalLayer(8)
tacl = ConvolutionalLayer(8)
mpc = MaxPoolingCalc()
trpl = PoolingLayer(3, 3, 2, mpc)
tapl = PoolingLayer(3, 3, 2, MaxPoolingCalc())
trfl = FlatteningLayer()
tafl = FlatteningLayer()
trfcl = FullyConnectedLayer(320,4, AdamWeightUpdateCalc())
trfcl2 = FullyConnectedLayer(160,4, AdamWeightUpdateCalc())
tafcl = FullyConnectedLayer(320,4, AdamWeightUpdateCalc())
tafcl2 = FullyConnectedLayer(160, 4, AdamWeightUpdateCalc())
trlsl = LogisticSigmoidLayer()
talsl = LogisticSigmoidLayer()
sal = SoftmaxActivationLayer()
cel = CrossEntropyLayer()

#Alternate activation and objective funcs (DeepMind paper used)
trrll1 = ReLuLayer()
tarll1 = ReLuLayer()
trrll2 = ReLuLayer()
tarll2 = ReLuLayer()
trrll3 = ReLuLayer()
tarll3 = ReLuLayer()
trsel = SquaredErrorLayer()
tasel = SquaredErrorLayer()

#Set up second kernel layer
trcl.setKernels(np.random.uniform(low=-pow(10,-4), high=pow(10,-4), size=(16, 3, 3)))
tacl.setKernels(np.random.uniform(low=-pow(10,-4), high=pow(10,-4), size=(16, 3, 3)))

trainingLayers = [tril, trcl, trpl, trrll1, trfl, trfcl, trrll2, trsel]
targetLayers = [tail, tacl, tapl, tarll1, tafl, tafcl, tarll2, tasel]

#Training model used for training and current q; used for action predictions
trainingModel = Model(trainingLayers)
trainingModel.load("2023-03-18-12-54-51_FC_1_CONV_1/2023-03-18-15-44-20_FC_1_CONV_1.npy")

#Target model used for future q and final model; more stable model to be updated
targetModel = Model(targetLayers)
targetModel.load("2023-03-18-12-54-51_FC_1_CONV_1/2023-03-18-15-44-20_FC_1_CONV_1.npy")

'''DEBUG CODE --- TO DELETE
Y = np.array([np.random.randint(8, size=4), np.random.randint(8, size=4)])
convLayerTrainingModelWeights, numKernelsPerObs, fcLayerTrainingModelWeights = trainingModel.getWeights()
targetModel.setWeights(convLayerTrainingModelWeights, numKernelsPerObs, fcLayerTrainingModelWeights)
trainingModel.train(X, Y)'''

#Make a prediction
prediction = trainingModel.predict(X)
#trainingModel.train(X,Y)
maxValIdx = np.argmax(prediction)

#Set action with predicted action
predictedAction = ActionTuple(np.zeros((1,0)), np.array([[maxValIdx]]))

#Create channel to specify run speed
channel = EngineConfigurationChannel()

#Open pacman environment
env = UE(file_name='./MiniGameMap/Pacman', seed=1, side_channels=[channel])

#Set game environment's run speed
channel.set_configuration_parameters(time_scale= 5.0)

env.reset()

#Get the name of the behavior we're using
behaviorName = list(env.behavior_specs)[0]

#Get the behavior spec which contains observation data
spec = env.behavior_specs[behaviorName]

#spec.action_spec is a ActionSpec tuple containing info on type of agent action and other action info
agentActionSpec = spec.action_spec

#Set action to predicted action
env.set_actions(behaviorName, predictedAction)

env.reset()

#Gets DecisionSteps object and TerminalSteps object
decisionSteps, terminalSteps = env.get_steps(behaviorName)

#Get the map state representation and reorder to grid size for CNN
decisionStepsObs = decisionSteps[0].obs
observation = np.array([np.reshape(decisionStepsObs, (NUM_MAP_ROWS, NUM_MAP_COLS))])

#Open a file to write episode rewards to for logging
logRoot = f'./TrainingScripts/Logs/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
os.makedirs(logRoot)
logFileName = 'episodeResults.csv'
resultsLogFile = open(f'{logRoot}/{logFileName}', 'w')
print("Episode,Epsilon,Rewards,QValMean", file=resultsLogFile)
resultsLogFile.close()

#Setting some loop params
trackedAgent = -1
done = False
episodeRewards = 0
#RHS returns the id of the agent
trackedAgent = decisionSteps.agent_id[0]
stepsToUpdateTargetModel = 0

for episode in range(NUM_TRAINING_EPISODES):
    if episode % 10 == 0:
        print(f'Training Episode: {episode}/{NUM_TRAINING_EPISODES}')
    episodeRewards = 0
    trackedAgent = -1
    done = False

    env.reset()

    #Gets DecisionSteps object and TerminalSteps object
    decisionSteps, terminalSteps = env.get_steps(behaviorName)

    if len(decisionSteps) >= 1:
        trackedAgent = decisionSteps.agent_id[0]
    
    assert trackedAgent != -1, "Error - no agent id set --- this shouldn't happen"

    #Get the map state representation and reorder to grid size for CNN
    decisionStepsObs = decisionSteps[trackedAgent].obs
    observation = np.array([np.reshape(decisionStepsObs, (NUM_MAP_ROWS, NUM_MAP_COLS))])

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
            if lastStepReward != 0 and NORMALIZE_REWARDS:
                lastStepReward = 1 if lastStepReward >= 1 else -1
            elif USE_LIVING_REWARD:
                lastStepReward = LIVING_REWARD
            episodeRewards += lastStepReward

        if trackedAgent in terminalSteps:
            lastStepReward = terminalSteps[trackedAgent].reward
            if lastStepReward != 0 and NORMALIZE_REWARDS:
                lastStepReward = 1 if lastStepReward >= 1 else -1
            elif USE_LIVING_REWARD:
                lastStepReward = LIVING_REWARD
            episodeRewards += lastStepReward
            done = True

        #Skip the training stuff every x episodes so we can just watch the model run normally
        #if (episode == 0) or (not episode % 5 == 0):
        if not done:
            decisionStepsObs = decisionSteps[trackedAgent].obs
            newObservation = np.array([np.reshape(decisionStepsObs, (NUM_MAP_ROWS, NUM_MAP_COLS))])
        else:
            newObservation = observation

        replayMemory[toAddReplayMemIdx] = np.array([observation, action, lastStepReward, newObservation, done])

        toAddReplayMemIdx += 1
        numAddedReplayMems += 1

        observation = newObservation

        #Update Training Model using Bellman Equation
        if stepsToUpdateTargetModel % NUM_STEPS_TILL_TRAIN == 0 or done:
            numAddedReplayMems = train(replayMemory, trainingModel, targetModel, numAddedReplayMems, done)
            if numAddedReplayMems < toAddReplayMemIdx:
                toAddReplayMemIdx = numAddedReplayMems

        #End episode every X steps
        if stepsToUpdateTargetModel >= NUM_STEPS_TILL_END_EPISODE:
            break

        if toAddReplayMemIdx == MAX_REPLAYMEMORY_SIZE:
            numAddedReplayMems = clearFirstXReplayMemories(numAddedReplayMems)
            if numAddedReplayMems < toAddReplayMemIdx:
                toAddReplayMemIdx = numAddedReplayMems

    # Update the weights of the target model w/ the weights of the training model ever X episodes
    if (episode + 1) % NUM_EPISODES_TILL_TRAIN_TARGET == 0:
        convLayerTrainingModelWeights, numKernelsPerObs, fcLayerTrainingModelWeights = trainingModel.getWeights()
        targetModel.setWeights(convLayerTrainingModelWeights, numKernelsPerObs, fcLayerTrainingModelWeights)
    stepsToUpdateTargetModel = 0

    #Update epsilon so we're less likely to take random action
    EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY * episode)
    
    #Get random batch for logging average q
    avgQBatch = np.zeros((REPLAY_MEMORY_BATCH_SIZE, replayMemory[0][0].shape[1], replayMemory[0][0].shape[2]))    
    for idx, memory in enumerate(replayMemory):
        if idx >= REPLAY_MEMORY_BATCH_SIZE:
            break        
        avgQBatch[idx] = memory[0]
    
    #Predict and save average Q
    qVals = trainingModel.predict(avgQBatch)
    qVals = np.max(qVals, axis=1)
    qValsMean = np.mean(qVals, axis=0)
    qValsMean = np.sum(qValsMean) / qValsMean.size

    #Clear some replay memory if full for next run
    if toAddReplayMemIdx == MAX_REPLAYMEMORY_SIZE:
        numAddedReplayMems = clearFirstXReplayMemories(numAddedReplayMems)
        if numAddedReplayMems < toAddReplayMemIdx:
            toAddReplayMemIdx = numAddedReplayMems   
    
    #Log Results
    resultsLogFile = open(f'{logRoot}/{logFileName}', 'a')
    print(str(episode) + "," + str(EPSILON) + "," + str(episodeRewards) + "," + str(qValsMean), file=resultsLogFile)
    if (USE_CUSTOM_FILE_PATH):
        trainingModel.serialize("NPY_FILES/")
    else:
        trainingModel.serialize()
    resultsLogFile.close()

# plot results
plot_metrics(logRoot, logFileName, trainingModel)

#Close environemnt when done
env.close()