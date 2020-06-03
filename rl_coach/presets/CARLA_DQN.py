from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.environments.carla_environment import CarlaEnvironmentParameters

from rl_coach.graph_managers.graph_manager import SimpleSchedule
from rl_coach.core_types import EnvironmentSteps
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.schedules import LinearSchedule

#########
# Agent #
#########
agent_params = DQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.00025
# rename the input embedder key from 'observation' to 'measurements'
agent_params.network_wrappers['main'].input_embedders_parameters['measurements'] = \
    agent_params.network_wrappers['main'].input_embedders_parameters.pop('observation')
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
agent_params.exploration.epsilon_schedule = LinearSchedule(1, 0.05, 300000) # 3000 episodes * 100 steps (1 step = 0.1s, timeout = 10s, so 1 episode = 100 steps)
agent_params.exploration.evaluation_epsilon = 0.05





####################
# Graph Scheduling #
####################
schedule_params = SimpleSchedule()
schedule_params.heatup_steps = EnvironmentSteps(20)

vis_params = VisualizationParameters(render=False)

###############
# Environment #
###############
env_params = CarlaEnvironmentParameters()

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
