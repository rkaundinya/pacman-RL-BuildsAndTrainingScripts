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

#Weights To Load Filename
toLoad =  "2023-03-18-18-18-17_FC_1_CONV_1/2023-03-18-18-56-55_FC_1_CONV_1.npy"

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

trcl.setKernels(np.random.uniform(low=-pow(10,-4), high=pow(10,-4), size=(16, 3, 3)))

testModelLayers = [tril, trcl, trpl, trrll1, trfl, trfcl, trrll2, trsel]

testModel = Model(testModelLayers)
testModel.load(toLoad)

#Create channel to specify run speed
channel = EngineConfigurationChannel()

#Open pacman environment
env = UE(file_name='../MiniGameMap/Pacman', seed=1, side_channels=[channel])

#Set environment run timescale
channel.set_configuration_parameters(time_scale= 1)
env.reset()
env.reset()

#Get the name of the behavior we're using
behaviorName = list(env.behavior_specs)[0]
#Get the behavior spec which contains observation data
spec = env.behavior_specs[behaviorName]

#spec.action_spec is a ActionSpec tuple containing info 
#on type of agent action and other action info
agentActionSpec = spec.action_spec

episodeRewards = 0
trackedAgent = -1
done = False

decisionSteps, terminalSteps = env.get_steps(behaviorName)
if len(decisionSteps) >= 1:
        trackedAgent = decisionSteps.agent_id[0]

assert trackedAgent != -1, "Error - no agent id set --- this shouldn't happen"

#Get the map state representation and reorder to grid size for CNN
decisionStepsObs = decisionSteps[trackedAgent].obs
observation = np.array([np.reshape(decisionStepsObs, (NUM_MAP_ROWS, NUM_MAP_COLS))])

while not done:
    if trackedAgent == -1 and len(decisionSteps) >= 1:
        trackedAgent = decisionSteps.agent_id[0]

    randActionProb = np.random.rand()

    prediction = testModel.predict(observation)
    maxValIdx = np.argmax(prediction)
    action = ActionTuple(np.zeros((1,0)), np.array([[maxValIdx]]))
    
    #Set action
    env.set_actions(behaviorName, action)
    env.step()

    decisionSteps, terminalSteps = env.get_steps(behaviorName)
    lastStepReward = 0
    if trackedAgent in decisionSteps:
        lastStepReward = decisionSteps[trackedAgent].reward
    if trackedAgent in terminalSteps:
        lastStepReward = terminalSteps[trackedAgent].reward
        done = True
    
    episodeRewards += lastStepReward

    #Just for debugging, track the observed states
    if not done:
        decisionStepsObs = decisionSteps[trackedAgent].obs
        newObservation = np.array([np.reshape(decisionStepsObs, (NUM_MAP_ROWS, NUM_MAP_COLS))])
    else:
        newObservation = observation

    observation = newObservation

print(episodeRewards)
#Close environment when done
env.close()