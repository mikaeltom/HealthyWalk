"""
Mikael Tom
9 June 2026
File that contains the environment created
for the Reinforcement Learning project.
"""
import random
import gymnasium as gym
import stable_baselines3 as sb3
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
import json
import time
from utils import DisplayType
import math


class HealthyWalkEnv(gym.Env):
    """
    Class representing the environment designed for the project about Ulqin, Montenegro. All details
    are explained in the report. It proposes segments of the graph of walks to the agent,
    that can either accept or reject. The environment contains an observation space of with 15 elements :
    user required time, the current loop  time, the time with the potential proposed node, ratio of
    nature, must have been places, amenity, must avoid places and pedestrian zones for the path
    to the proposed node. Also, there are normalized coordinates about the last chosen node, the potential node,
    the guiding must have been place (for reward), shortest distance to the source node and if the proposed node
    is a dead end. It starts at a fixed position  with node id 2340203954. It contains a reward function that
    penalizes or reward segment choices.
    """
    def __init__(self, display=DisplayType.NO_DISPLAY):
        self.episode_count = 0 # for displaying purpose in GUI
        self.current_map_to_display = None # for displaying purpose in GUI
        self.asking_rating = False # for displaying purpose in GUI
        self.current_time_to_display = None # for displaying purpose in GUI
        self.display_behaviour = display.value # for display purpose
        self.received_rating = 0

        self.location = "Ulcinj, Montenegro"
        self.start_location = 2340203954
        self.graphOfWalks = ox.graph.graph_from_place(self.location, network_type="walk")
        self.observation_space = gym.spaces.Box(low=np.array([0.0] * 16, dtype=np.float32), high=np.array([1.0] * 16,
                                                                    dtype=np.float32), shape=(16,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2) # accept or reject the proposed portion of path
        self.min_x, self.max_x, self.min_y, self.max_y = self.get_min_max_coordinates()
        self.deadends_location = self.find_deadends()
        self.natural_places, self.must_been_places, self.amenity_places, self.must_avoid_places, self.pedestrian_places = self.load_json_locations()
        self.direction_to_must_been_places = random.choice(list(self.must_been_places)) # node used to direct the agent to the must been place
        self.natural_ratio, self.must_been_ratio, self.amenity_ratio, self.must_avoid_ratio, self.pedestrian_ratio = 0.0, 0.0, 0.0, 0.0, 0.0
        self.set_base_values()

    def set_base_values(self):
        """
        Method that sets/defines all the instances/variables at the start of the episode.
        """
        self.last_index = 0 # index of the
        self.current_total_distance = 0 # distance of the current loop that the agent is creating
        self.last_location = self.start_location # last picked node
        self.current_loop = [self.start_location, self.last_location] # the nodes that are in the agent's created loop
        self.time = np.random.uniform(5, 50) # simulate the requested time by user
        self.potential_location = self.generate_proposed_node() # compute the proposed new node
        self.future_distance = self.compute_potential_loop_distance() # compute loop distance with this potential node
        self.compute_potential_quality() # compute the 5 categories ratio of the proposed loop


    def reset(self, **kwargs):
        """
        Method that resets the environment at the end of the episode.
        """
        self.set_base_values()
        return self.make_state(), {}


    def generate_proposed_node(self):
        """
        Function that generates the proposed node to the agent. The agent will observe
        the path to this node and make a decision (accept or reject). This randomly proposed node
        is found using the following strategy : If less than 80% of the time is used, it encourages
        the agent to explore node that are in a radius between 500m and 1000m from the last picked node.
        As the walk progress toward 80%, the radius is lower : 300m to 500m. If more than 80% percent of
        the time is used, or if the requested time is low (less than 15 minutes walk), it pick a random
        node within 300m radius. In both cases, it clean the duplicates nodes.
        """
        current_time = self.convert_meter_to_minutes(self.current_total_distance) / self.time
        if self.time >= 15 and current_time < 0.8:
            low, up = (500, 1001) if current_time < 0.5 else (300, 501)
            neighbors = nx.ego_graph(self.graphOfWalks, self.last_location, radius=up, distance="length") # source = https://networkx.org/documentation/stable/auto_examples/drawing/plot_ego_graph.html
            low_neighbors = nx.ego_graph(self.graphOfWalks, self.last_location, radius=low, distance="length")
            neighbors_list = set(neighbors.nodes) - set(low_neighbors.nodes) # remove nodes below
        else:
            neighbors = nx.ego_graph(self.graphOfWalks, self.last_location, radius=300, distance="length")
            neighbors_list = set(neighbors.nodes)
        potential_nodes = list(neighbors_list - (set(self.current_loop) - {self.last_location})) # clean duplicates nodes
        if len(potential_nodes) <= 1: # check, if after clean has some nodes, otherwise, remove the duplicate cleaning
            potential_nodes = list(neighbors.nodes) # there is no other option than picking a duplicate in such case
        random_node_proposed = np.random.choice(potential_nodes)
        return random_node_proposed


    def perform_accepted(self):
        """
        Method that performs the action of accepting a proposed node.
        """
        new_locations = nx.shortest_path(self.graphOfWalks, self.last_location, self.potential_location, weight="length")
        self.last_location = self.potential_location
        self.current_loop = self.current_loop[:self.last_index] + new_locations[:]
        self.last_index = len(self.current_loop) - 1
        end_the_loops_locations = nx.shortest_path(self.graphOfWalks, self.last_location, self.start_location,weight="length")
        self.current_loop += end_the_loops_locations[1:]
        if len(self.current_loop) > 2: # to avoid computing for nothing when only 2 elements
            self.current_total_distance = self.compute_loop_distance(self.current_loop)


    def step(self, action):
        """
        Function that performs one step of the agent. If he accepted the path,
        the current loop is updated. Then, it verifies if the episode must be
        set to finished. If it is the case it handles the display for the GUI.
        Otherwise, it generates another proposed node. Finally, it computes
        the reward and make the current state.
        """
        is_accepted = action == 1
        phi_st = self.get_potential()

        if is_accepted:
            self.perform_accepted()

        done, rating = self.verify_done()

        if done:
            self.current_map_to_display = self.current_loop
        else:
            self.potential_location = self.generate_proposed_node()
            self.future_distance = self.compute_potential_loop_distance()
            self.compute_potential_quality()

        reward = self.reward_function(is_accepted, done, rating, phi_st)
        return self.make_state(), reward, done, False, {}


    def verify_done(self):
        """
        Method that indicates if the episode must be terminated. There are two ways to end an episode:
        The current time exceeds the required time, or, the current time is two minutes less than
        required time. We assume that having two minutes less is a realistic and suited difference as
        creating path with exactly an equivalent time is challenging for the agent as distance between nodes
        is not equivalent, thus being precise is such condition is not always possible. It also handles
        the update of information to display (if in the GUI mode) and the rating retrieval.
        """
        if (self.convert_meter_to_minutes(self.current_total_distance) > self.time) or self.has_two_minutes_less():
            if self.display_behaviour == DisplayType.SIMPLE_DISPLAY.value or self.display_behaviour == DisplayType.RATING_DISPLAY.value:
                rating = self.display_loop()
            else:
                rating = None
            return True, rating
        return False, None


    def reward_function(self, is_accepted, done, rating, phi_st):
        """
        Reward function. It sends positive and negative rewards regarding the accepted path.
        The natural, must have been, pedestrian, amenity ratios give positive rewards, while
        must avoid ratio give a strong negative reward. In addition, it also sends a negative
        reward if the picked node is a dead end. If the path was not accepted, it sends a slight
        negative reward to ensure that the agent does not infinitly reject paths. it also adds
        the user rating reward (optional). A negative reward is given if the walk is longer than
        the requested time, and a positive reward is given depending if it is 1 minute or 2 minute close.
        Finally, there is potential based reward on the distance to the must have been places,
        to guide the agent to them.
        """
        score = 0.0
        if is_accepted:
            score += self.natural_ratio
            score += self.must_been_ratio * 4 # important to visit them
            score += self.pedestrian_ratio
            score += self.amenity_ratio / 2 # there are a lot so not that important
            score -= self.must_avoid_ratio * 4 # dangerous must really be avoided for security
            if self.last_location in self.deadends_location:
                score -= 1.0
        else: # reject
            score -= 0.1 # negative reward when reject to avoid infinite reject

        if rating is not None:
            rating_map = [-1, -0.5, 1, 2, 4] # which star is mapped to what
            score += rating_map[rating-1]

        if done :
            time = self.convert_meter_to_minutes(self.current_total_distance)
            if time > self.time:
                score -= (((time) / self.time) - 1 ) * 4
            if time <= self.time and self.has_two_minutes_less(): # if two minutes close
                score += (time / self.time)
            if abs(time - self.time) <= 1: # we consider 1 as perfect timing
                score += 1.0

        phi_stplus1 = self.get_potential()
        score += 0.99 * phi_st - phi_stplus1
        return score


    def make_state(self):
        """
        Method that create the current state s during the episode. The agent can observe multiple things :
        It contains the user required time, the current loop time, the time with the potential proposed node.
        These 3 times are normalized by dividing them by 70. Even if the system is limited between
        5 and 50 minutes, we chose 70 to have a safety margin to not exceed 1. Then there is the ratio of
        nature, must have been places, amenity, must avoid places and pedestrian zones for the path
        to the proposed node.Then, there are normalized coordinates about the last chosen node, the potential node,
        the guiding must have been place (for reward). It also contains the shortest distance to the source node and
        observe if the proposed node is a dead end.
        """
        x_last, y_last = self.get_normalized_coordinates(self.last_location)
        x_pot, y_pot = self.get_normalized_coordinates(self.potential_location)
        x_must_been, y_must_been = self.get_normalized_coordinates(self.direction_to_must_been_places)
        dist_to_start = nx.shortest_path_length(self.graphOfWalks, self.potential_location, self.start_location,weight="length")
        dist_to_start = dist_to_start / 5833 # 5833 because we can max do 50 minutes walk BUT assume that you can do 70 minutes by mistake, so this is the distance of 5km/h for 70minutes
        is_deadend = float(self.potential_location in self.deadends_location)
        return np.array(
            [self.time / 70, self.convert_meter_to_minutes(self.current_total_distance) / 70, self.convert_meter_to_minutes(self.future_distance) / 70,
             self.natural_ratio, self.must_been_ratio, self.amenity_ratio, self.must_avoid_ratio, self.pedestrian_ratio, x_last, y_last, x_pot, y_pot, x_must_been, y_must_been, dist_to_start, is_deadend],
            dtype=np.float32)


    def get_normalized_coordinates(self, node_id):
        """
        Method that normalizes in coordinates(x,y) both x and y between 0 an 1.
        We did this by using the classical min-max normalization :https://en.wikipedia.org/wiki/Feature_scaling
        """
        node = self.graphOfWalks.nodes[node_id]
        x, y = node['x'], node['y'] # source : https://stackoverflow.com/questions/46238813/osmnx-get-coordinates-of-nodes-using-osm-id
        x = (x - self.min_x) / (self.max_x - self.min_x) # source : https://stackoverflow.com/questions/48178884/min-max-normalisation-of-a-numpy-array
        y = (y - self.min_y) / (self.max_y - self.min_y)
        return x, y


    def get_potential(self):
        """
        Method used for the potential based reward (reward shaping). It returns a normalized walking
        time to the must have been places representative selected at the beginning of the episode.
        """
        x,y = self.get_normalized_coordinates(self.last_location)
        must_x, must_y = self.get_normalized_coordinates(self.direction_to_must_been_places)
        meter = math.sqrt((must_x - x)**2 + (must_y - y)**2)
        return self.convert_meter_to_minutes(meter) #/ self.time


    def has_two_minutes_less(self):
        """
        Function verifying if there are two minutes or less left for the walk.
        """
        current_minutes = self.convert_meter_to_minutes(self.current_total_distance)
        return 0 <= self.time - current_minutes <= 2


    def compute_potential_loop_distance(self):
        """
        Function that computes the distance of the proposed loop containing the potential node that will be added.
        As explained in the report, the path between the last node and the source node (start) is truncated
        (using the saved index) and replaced by the path between this potential node and the source.
        """
        new_locations = nx.shortest_path(self.graphOfWalks, self.last_location, self.potential_location, weight="length")
        current_loop = self.current_loop[:self.last_index] + new_locations[:]
        end_the_loops_locations = nx.shortest_path(self.graphOfWalks, self.potential_location, self.start_location, weight="length")
        current_loop += end_the_loops_locations[1:]
        return self.compute_loop_distance(current_loop)


    def compute_loop_distance(self, current_loop):
        """
        Given a set of nodes representing the current loop. It computes the total distance
        by summing the shortest path length between consecutive nodes.
        """
        total = 0
        size = len(current_loop)
        for i in range(size - 1):
            start = current_loop[i]
            arrival = current_loop[i + 1]
            shortest_path = nx.shortest_path_length(self.graphOfWalks, start, arrival, weight="length")
            total += shortest_path
        return total


    def get_min_max_coordinates(self):
        """
        Method that retrieve the min, max for lat (x) and lon (y) in the graph.
        This will be used for min-max normalization of coordinates.
        """
        nodes = ox.graph_to_gdfs(self.graphOfWalks, edges=False) # source : https://www.timlrx.com/blog/cleaning-openstreetmap-intersections-in-python/
        list_x, list_y = nodes['x'], nodes['y'] # source : https://www.timlrx.com/blog/cleaning-openstreetmap-intersections-in-python/
        min_x, max_x = list_x.min(), list_x.max() # source : https://www.geeksforgeeks.org/pandas/python-pandas-series-min/
        min_y, max_y = list_y.min(), list_y.max()
        return min_x, max_x, min_y, max_y


    def convert_meter_to_minutes(self, m):
        """
        Convert meter in km then in minutes.
        We assume that the user walk at 5km/h.
        """
        km = m / 1000
        return km * 12


    def compute_ratio(self, potential_locations, tags_locations):
        """
        Computes a ratio, linked to the 5 categories, for the potential path that will be added.
        The idea is
        """
        size = len(potential_locations)
        return len(set(potential_locations).intersection(tags_locations)) / size


    def compute_potential_quality(self):
        """
        As explained in the report, we identified what are the components necessary for a "good walk". Based on the work
        Saelens, B. E., & Handy, S. L. (2008). Built environment correlates of walking: A review. Medicine & Science in
        Sports & Exercise, 40 (7), S550–S566. https://doi.org/10.1249/MSS.0b013e31817c67a4
        We have different ratio indicating different ideas of what is a good walk :
        - natural ratio = amount of nodes with nature linked tags
        - must have been ratio = nodes with monuments or historical places tags
        - amenity ratio = nodes with amenity tags
        - pedestrian ratio = nodes with pedestrian safety tags (pedestrian zone, low maximum speed, etc.)
        - must avoid ratio = nodes not safe for pedestrians (high speed, lots of traffic, etc.)
        More details about each are described in the report.
        The agent receive the proposed road and
        """
        potential_locations = nx.shortest_path(self.graphOfWalks, self.last_location, self.potential_location, weight="length")
        self.natural_ratio = self.compute_ratio(potential_locations, self.natural_places)
        self.must_been_ratio = self.compute_ratio(potential_locations, self.must_been_places)
        self.amenity_ratio = self.compute_ratio(potential_locations, self.amenity_places)
        self.must_avoid_ratio = self.compute_ratio(potential_locations, self.must_avoid_places)
        self.pedestrian_ratio = self.compute_ratio(potential_locations, self.pedestrian_places)


    def find_deadends(self):
        """
        Methods that find the dead ends. It is used before the start of the episode.
        The idea is that dead ends are bad, as you must return to your previous steps.
        For this reason avoiding them makes your walk more smoothly without repetitions.
        """
        streets = ox.stats.count_streets_per_node(self.graphOfWalks) # source : https://www.timlrx.com/blog/cleaning-openstreetmap-intersections-in-python/
        list_of_deads = []
        for x, y in streets.items():
            if y == 1: # check if count is 1, keep only the nodes representing a single street (= dead ends)
                list_of_deads.append(x) # adding the node id
        return list_of_deads


    def display_loop(self):
        """
        Function used only for the GUI mode. It handles the update of required information
        for the GUi such as the end episode loop and its time. In addition, the rating reception
        is handled here. The idea is that asking_rating is changed by the GUI when the user click on
        the rating related buttons. Otherwise, it stays in an infinite loop untile the user answers.
        """
        self.episode_count += 1
        if self.display_behaviour == DisplayType.SIMPLE_DISPLAY.value or self.display_behaviour == DisplayType.RATING_DISPLAY.value:
            self.current_map_to_display = self.current_loop
            self.current_time_to_display = (self.time, self.convert_meter_to_minutes(self.current_total_distance))
            if  self.display_behaviour == DisplayType.RATING_DISPLAY.value and self.episode_count % 100 == 0: # ask rating only every 100 episode
                self.asking_rating = True
                while self.asking_rating: # infinite loop that is 'broke' by the GUI functions
                    time.sleep(0.01) # the idea is that it will change self.asking_rating when rating is given by user
                rating = self.received_rating
                self.received_rating = None
                return rating


    def load_json_locations(self):
        """
        Retrieve from the json files of tags, the nodes from each of the categories.
        """
        with open("tags_locations.json", "r") as f:
            data = json.load(f)
            return set(data["natural"]), set(data["must_been"]), set(data["amenity"]), set(data["must_avoid"]), set(data["pedestrian"])
