from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import carla

from rl_coach.environments.carla.render import BirdeyeRender
from rl_coach.environments.carla.route_planner import RoutePlanner
from rl_coach.environments.carla.misc import *

# Coach imports
from rl_coach.logger import screen
from rl_coach.filters.action.partial_discrete_action_space_map import PartialDiscreteActionSpaceMap
from rl_coach.filters.observation.observation_rgb_to_y_filter import ObservationRGBToYFilter
from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter
import os
import signal
import logging
import subprocess
from rl_coach.environments.environment import Environment, EnvironmentParameters, LevelSelection
from rl_coach.spaces import BoxActionSpace, DiscreteActionSpace, ImageObservationSpace, StateSpace, VectorObservationSpace, PlanarMapsObservationSpace
from rl_coach.utils import get_open_port, force_list
from enum import Enum
from typing import List, Union
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from rl_coach.filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter
from rl_coach.filters.observation.observation_stacking_filter import ObservationStackingFilter


CARLA_WEATHER_PRESETS = {
    0: carla.WeatherParameters.Default,
    1: carla.WeatherParameters.ClearNoon,
    2: carla.WeatherParameters.CloudyNoon,
    3: carla.WeatherParameters.WetNoon,
    4: carla.WeatherParameters.WetCloudyNoon,
    5: carla.WeatherParameters.MidRainyNoon,
    6: carla.WeatherParameters.HardRainNoon,
    7: carla.WeatherParameters.SoftRainNoon,
    8: carla.WeatherParameters.ClearSunset,
    9: carla.WeatherParameters.CloudySunset,
    10: carla.WeatherParameters.WetSunset,
    11: carla.WeatherParameters.WetCloudySunset,
    12: carla.WeatherParameters.MidRainSunset,
    13: carla.WeatherParameters.HardRainSunset,
    14: carla.WeatherParameters.SoftRainSunset
}

# Set up the input and output filters
# Input filters apply operations to the input space defined, several input filters can be defined. The order is executed sequentially
# Output filters apply operations on the output space defined.
CarlaInputFilter = NoInputFilter()
CarlaOutputFilter = NoOutputFilter()

# Enumerate observation sources
# New observation sources need to be appended here
class SensorTypes(Enum):
    FRONT_CAMERA = "forward_camera"
    SEMANTIC = "semseg_camera"
    LIDAR = "lidar"
    BIRDEYE = "birdeye"

class CarlaEnvironmentParameters(EnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.host = 'localhost'
        self.port = 2000
        self.timeout = 10.0 # Carla client timeout
        self.level = 'Town03' # Name of the world
        self.number_of_vehicles = 0 # traffic
        self.number_of_walkers = 0 # pedestrian traffic
        self.weather_id = 1 # Weather IDs: https://carla.readthedocs.io/en/stable/carla_settings/

        self.frame_skip = 1 # number of frames to repeat the same action

        self.ego_vehicle_filter = 'vehicle.lincoln*' # filter for defining ego vehicle
        self.display_size = 256 # screen size of bird-eye render
        self.display_route = True # whether to render the desired route
        self.render_pygame = True # whether to render the pygame window
        # Example of adding sensors: self.sensors = [SensorTypes.FRONT_CAMERA, SensorTypes.LIDAR]
        self.sensors = [] # defines a list of sensors for the state space
        self.rgb_camera_height = 256
        self.rgb_camera_width = 256
        self.semseg_camera_height = 256
        self.semseg_camera_width = 256
        self.obs_range = 32 # observation range (meter)
        self.lidar_bin = 0.125 # bin size of lidar sensor (meter)
        self.d_behind = 12 # distance behind the ego vehicle (meter)
        self.out_lane_thres = 2.0 # threshold for out of lane
        self.desired_speed = 8 # desired speed (m/s)

        self.discrete = True # whether to use discrete control space
        self.discrete_acc = [-3.0, -1.5, 0.0, 1.5, 3.0] # discrete value of accelerations
        self.discrete_steer = [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9] # discrete value of steering angles
        self.continuous_accel_range = [-3.0, 3.0]  # continuous acceleration range
        self.continuous_steer_range = [-0.3, 0.3]  # continuous steering angle range

        self.max_past_step = 1 # the number of past steps to draw
        self.dt = 0.1  # time interval between two frames
        self.max_ego_spawn_times = 200 # maximum times to spawn ego vehicle
        self.max_time_episode = 300 # maximum `timestep`s per episode
        self.max_waypt = 12 # maximum number of waypoints

        self.default_input_filter = CarlaInputFilter
        self.default_output_filter = CarlaOutputFilter

    @property
    def path(self):
        return 'rl_coach.environments.carla_environment:CarlaEnvironment'

class CarlaEnvironment(Environment):
    """A coach wrapper for CARLA simulator."""
    def __init__(self, level: LevelSelection,
               seed: int, frame_skip: int, human_control: bool, custom_reward_threshold: Union[int, float],
               visualization_parameters: VisualizationParameters,
               host: str, port: int, timeout: float,
               number_of_vehicles: int, number_of_walkers: int, weather_id: int, #rendering_mode: bool,
               ego_vehicle_filter: str, display_size: int,
               sensors: List[SensorTypes], rgb_camera_height: int, rgb_camera_width: int,
               semseg_camera_height: int, semseg_camera_width: int,
               lidar_bin: float, obs_range: float, display_route: bool, render_pygame: bool,
               d_behind: float, out_lane_thres: float, desired_speed: float, max_past_step: int,
               dt: float, discrete: bool, discrete_acc: List[float], discrete_steer: List[float],
               continuous_accel_range: List[float], continuous_steer_range: List[float],
               max_ego_spawn_times: int, max_time_episode: int, max_waypt: int, **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters)

        self.level = level
        # self.frame_skip = frame_skip
        # self.seed = seed
        # self.human_control = human_control
        # self.custom_reward_threshold = custom_reward_threshold
        # self.visualization_paramters = visualization_parameters

        self.host = host
        self.port = port
        self.timeout = timeout
        self.number_of_vehicles = number_of_vehicles
        self.number_of_walkers = number_of_walkers
        self.weather_id = weather_id

        self.ego_vehicle_filter = ego_vehicle_filter
        self.display_size = display_size
        self.sensors = sensors
        self.rgb_camera_height = rgb_camera_height
        self.rgb_camera_width = rgb_camera_width
        self.semseg_camera_height = semseg_camera_height
        self.semseg_camera_width = semseg_camera_width
        self.obs_range = obs_range
        self.lidar_bin = lidar_bin
        self.obs_size = int(self.obs_range/self.lidar_bin)
        self.display_route = display_route
        self.render_pygame = render_pygame
        self.d_behind = d_behind
        self.out_lane_thres = out_lane_thres
        self.desired_speed = desired_speed

        self.max_past_step = max_past_step
        self.dt = dt
        self.discrete = discrete
        self.discrete_acc = discrete_acc
        self.discrete_steer = discrete_steer
        self.continuous_accel_range = continuous_accel_range
        self.continuous_steer_range = continuous_steer_range
        self.max_ego_spawn_times = max_ego_spawn_times
        self.max_time_episode = max_time_episode
        self.max_waypt = max_waypt

        # Connect to carla server and get world object
        print('connecting to Carla server...')
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.traffic_manager = self.client.get_trafficmanager()
        self.world = self.client.load_world(level)
        print('Carla server connected!')

        # Set weather
        self.world.set_weather(CARLA_WEATHER_PRESETS[self.weather_id])

        # Get spawn points
        self._get_spawn_points()

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(self.ego_vehicle_filter, color='49,8,8')

        # Collision sensor
        self.collision_hist = [] # The collision history
        self.collision_hist_l = 1 # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Lidar sensor
        self.lidar_data = None
        self.lidar_height = 2.1
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', '32')
        self.lidar_bp.set_attribute('range', '5000')

        # Camera sensor
        self.rgb_camera_img = np.zeros((self.rgb_camera_height, self.rgb_camera_width, 3), dtype=np.uint8)
        self.rgb_camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.rgb_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.rgb_camera_bp.set_attribute('image_size_x', str(self.rgb_camera_width))
        self.rgb_camera_bp.set_attribute('image_size_y', str(self.rgb_camera_height))
        self.rgb_camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.rgb_camera_bp.set_attribute('sensor_tick', '0.02')

        # Semantic segmentation camera
        self.semseg_camera_img = np.zeros((self.semseg_camera_height, self.semseg_camera_width, 3), dtype=np.uint8)
        self.semseg_camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.semseg_camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        # Modify the attributes of the blueprint
        self.semseg_camera_bp.set_attribute('image_size_x', str(self.semseg_camera_width))
        self.semseg_camera_bp.set_attribute('image_size_y', str(self.semseg_camera_height))
        self.semseg_camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.semseg_camera_bp.set_attribute('sensor_tick', '0.02')

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt
        #self.settings.no_rendering_mode = rendering_mode
        self._set_synchronous_mode(True)

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # Action space
        if self.discrete:
            self.discrete_act = [discrete_acc, discrete_steer]
            self.n_acc = len(self.discrete_act[0])
            self.n_steer = len(self.discrete_act[1])
            self.action_space = DiscreteActionSpace(num_actions=self.n_acc*self.n_steer, descriptions=["acceleration", "steering"])
        else:
            self.action_space = BoxActionSpace(shape=2, low=np.array([continuous_accel_range[0], continuous_steer_range[0]]),
            high=np.array([continuous_accel_range[1], continuous_steer_range[1]]), descriptions=["acceleration", "steering"])

        # Observation space
        self.state_space = StateSpace({
            "measurements": VectorObservationSpace(shape=4, low=np.array([-2, -1, -5, 0]), high=np.array([2, 1, 30, 1]),
                                                    measurements_names=["lat_dist", "heading_error", "ego_speed", "safety_margin"])
        })

        if SensorTypes.FRONT_CAMERA in self.sensors:
            self.state_space[SensorTypes.FRONT_CAMERA.value] = ImageObservationSpace(shape=np.array([self.rgb_camera_height, self.rgb_camera_width, 3]), high=255)
        if SensorTypes.SEMANTIC in self.sensors:
            self.state_space[SensorTypes.SEMANTIC.value] = ImageObservationSpace(shape=np.array([self.semseg_camera_height, self.semseg_camera_width, 3]), high=255)
        if SensorTypes.LIDAR in self.sensors:
            self.state_space[SensorTypes.LIDAR.value] = PlanarMapsObservationSpace(shape=np.array([self.obs_size, self.obs_size, 3]), low=0, high=255)
        if SensorTypes.BIRDEYE in self.sensors:
            self.state_space[SensorTypes.BIRDEYE.value] = ImageObservationSpace(shape=np.array([self.obs_size, self.obs_size, 3]), high=255)

        # Initialize the renderer
        self._init_renderer()

        self.reset_internal_state(True)

    def _update_state(self):
        """ Update the internal state of the wrapper
        self.state - a dictionary containing all the observations from the environment and follows the state_space definition
        self.reward - float value containing the reward for the last step of the environment
        self.done - boolean flag which signals if the environment episode has ended
        self.goal - numpy array representing the goal the environment has set for the last step
        self.info - dictionary that contains any additional information for the last step
        """

        self.state = {}
        self.reward = None
        self.done = None

        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
          self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
          self.walker_polygons.pop(0)

        # route planner
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # state information
        info = {
          'waypoints': self.waypoints,
          'vehicle_front': self.vehicle_front
        }

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        self.state = self._get_obs()
        self.reward = self._get_reward()
        self.done = self._terminal()
        self.info = copy.deepcopy(info)

    def _take_action(self, action):
        """ Gets the action from the agent and makes a single step on the environment """
        # Calculate acceleration and steering
        if self.discrete:
            acc = self.discrete_act[0][action//self.n_steer]
            steer = self.discrete_act[1][action%self.n_steer]
        else:
            acc = action[0]
            steer = action[1]

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc/3,0,1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc/8,0,1)

        # Apply control
        act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        self.ego.apply_control(act)

        self.world.tick()

    def _restart_environment_episode(self, force_environment_reset=False):
        """ Restart the environment on a new episode """
        # Clear sensor objects
        self.collision_sensor = None
        self.lidar_sensor = None
        self.rgb_camera_sensor = None
        self.semseg_camera_sensor = None

        # Delete sensors, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'sensor.camera.semantic_segmentation', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

        # Disable sync mode
        self._set_synchronous_mode(False)

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1

        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                count -= 1

        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                self._restart_environment_episode()

            transform = random.choice(self.vehicle_spawn_points)
            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))
        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist)>self.collision_hist_l:
                self.collision_hist.pop(0)
        self.collision_hist = []

        # Add lidar sensor
        if SensorTypes.LIDAR in self.sensors:
            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
            self.lidar_sensor.listen(lambda data: get_lidar_data(data))
            def get_lidar_data(data):
                self.lidar_data = data

        # Add camera sensor
        if SensorTypes.FRONT_CAMERA in self.sensors:
            self.rgb_camera_sensor = self.world.spawn_actor(self.rgb_camera_bp, self.rgb_camera_trans, attach_to=self.ego)
            self.rgb_camera_sensor.listen(lambda data: get_rgb_img(data))
            def get_rgb_img(data):
                array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.rgb_camera_img = array

        # Add semantic segmentation sensor
        if SensorTypes.SEMANTIC in self.sensors:
            self.semseg_camera_sensor = self.world.spawn_actor(self.semseg_camera_bp, self.semseg_camera_trans, attach_to=self.ego)
            self.semseg_camera_sensor.listen(lambda data: get_semantic_img(data))
            def get_semantic_img(data):
                data.convert(carla.ColorConverter.CityScapesPalette)
                array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.semseg_camera_img = array


        # Update timesteps
        self.time_step=0
        self.reset_step+=1

        # Enable sync mode
        self._set_synchronous_mode(True)
        self.world.tick()

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # Set ego information for render
        self.birdeye_render.set_hero(self.ego, self.ego.id)

    def get_rendered_image(self) -> np.ndarray:
        return self.birdeye
        #return resize(self.rgb_camera_img, (self.rgb_camera_height, self.rgb_camera_width)) * 255

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
          bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _init_renderer(self):
        """Initialize the birdeye view renderer.
        """
        pygame.init()
        self.display = pygame.display.set_mode(
        (self.display_size * 4, self.display_size),
        pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
        birdeye_params = {
          'screen_size': [self.display_size, self.display_size],
          'pixels_per_meter': pixels_per_meter,
          'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

    def _set_synchronous_mode(self, synchronous = True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.traffic_manager.set_synchronous_mode(synchronous)
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot()
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego=vehicle
            return True

        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
          filt: the filter indicating what type of actors we'll look at.

        Returns:
          actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict={}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans=actor.get_transform()
            x=trans.location.x
            y=trans.location.y
            yaw=trans.rotation.yaw/180*np.pi
            # Get length and width
            bb=actor.bounding_box
            l=bb.extent.x
            w=bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            # Get global bounding box polygon
            poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
            actor_poly_dict[actor.id]=poly
        return actor_poly_dict

    def _get_obs(self):
        """Get the observations."""
        ## Birdeye rendering
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        self.birdeye_render.walker_polygons = self.walker_polygons
        self.birdeye_render.waypoints = self.waypoints

        # birdeye view with roadmap and actors
        birdeye_render_types = ['roadmap', 'actors']
        if self.display_route:
          birdeye_render_types.append('waypoints')
        self.birdeye_render.render(self.display, birdeye_render_types)
        birdeye = pygame.surfarray.array3d(self.display)
        birdeye = birdeye[0:self.display_size, :, :]
        birdeye = display_to_rgb(birdeye, self.obs_size)

        ## Lidar image generation
        if SensorTypes.LIDAR in self.sensors:
            point_cloud = []
            # Get point cloud data
            for location in self.lidar_data:
              point_cloud.append([location.x, location.y, -location.z])
            point_cloud = np.array(point_cloud)
            # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
            # and z is set to be two bins.
            y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind+self.lidar_bin, self.lidar_bin)
            x_bins = np.arange(-self.obs_range/2, self.obs_range/2+self.lidar_bin, self.lidar_bin)
            z_bins = [-self.lidar_height-1, -self.lidar_height+0.25, 1]
            # Get lidar image according to the bins
            lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
            lidar[:,:,0] = np.array(lidar[:,:,0]>0, dtype=np.uint8)
            lidar[:,:,1] = np.array(lidar[:,:,1]>0, dtype=np.uint8)
            # Add the waypoints to lidar image
            if self.display_route:
              wayptimg = (birdeye[:,:,0] <= 10) * (birdeye[:,:,1] <= 10) * (birdeye[:,:,2] >= 240)
            else:
              wayptimg = birdeye[:,:,0] < 0  # Equal to a zero matrix
            wayptimg = np.expand_dims(wayptimg, axis=2)
            wayptimg = np.fliplr(np.rot90(wayptimg, 3))

            # Get the final lidar image
            lidar = np.concatenate((lidar, wayptimg), axis=2)
            lidar = np.flip(lidar, axis=1)
            lidar = np.rot90(lidar, 1)
            lidar = lidar * 255

        ## Display camera image
        if SensorTypes.FRONT_CAMERA in self.sensors:
            camera = resize(self.rgb_camera_img, (self.rgb_camera_height, self.rgb_camera_width)) * 255

        ## Display semantic image
        if SensorTypes.SEMANTIC in self.sensors:
            semantic = resize(self.semseg_camera_img, (self.semseg_camera_height, self.semseg_camera_width)) * 255

        # State observation
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw/180*np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w,
          np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

        # Update the states in the state space
        obs = {}
        obs['measurements'] = state
        if SensorTypes.FRONT_CAMERA in self.sensors:
            obs[SensorTypes.FRONT_CAMERA.value] = camera.astype(np.uint8)
        if SensorTypes.SEMANTIC in self.sensors:
            obs[SensorTypes.SEMANTIC.value] = semantic.astype(np.uint8)
        if SensorTypes.LIDAR.value in self.sensors:
            obs[SensorTypes.LIDAR.value] = lidar.astype(np.uint8)
        if SensorTypes.BIRDEYE.value in self.sensors:
            obs[SensorTypes.BIRDEYE.value] = birdeye.astype(np.uint8)


        if self.render_pygame:
            # Display birdeye image
            self.birdeye = birdeye
            birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
            self.display.blit(birdeye_surface, (0, 0))

            # Display lidar image
            if SensorTypes.LIDAR in self.sensors:
                lidar_surface = rgb_to_display_surface(lidar, self.display_size)
                self.display.blit(lidar_surface, (self.display_size, 0))

            # Display camera image
            if SensorTypes.FRONT_CAMERA in self.sensors:
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size * 2, 0))

            # Display semantic image
            if SensorTypes.SEMANTIC in self.sensors:
                semantic_surface = rgb_to_display_surface(semantic, self.display_size)
                self.display.blit(semantic_surface, (self.display_size * 3, 0))

            # Display on pygame
            pygame.display.flip()

        return obs

    def _get_reward(self):
        """Calculate the step reward."""
        # reward for speed tracking
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        r_speed = -abs(speed - self.desired_speed)

        # reward for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1

        # reward for steering:
        r_steer = -self.ego.get_control().steer**2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > self.out_lane_thres:
            r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > self.desired_speed:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

        r = 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1

        return r

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist)>0:
            return True

        # If reach maximum timestep
        if self.time_step>self.max_time_episode:
            return True

        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres:
            return True

        return False

    def _get_spawn_points(self):
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
          spawn_point = carla.Transform()
          loc = self.world.get_random_location_from_navigation()
          if (loc != None):
            spawn_point.location = loc
            self.walker_spawn_points.append(spawn_point)

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            actor_list = self.world.get_actors().filter(actor_filter)
            # if actor_filter == 'controller.ai.walker' and actor_list is not None:
            #     for actor in actor_list:
            #         actor.stop()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
