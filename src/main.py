"""
Mikael Tom
9 June 2026
File that contains the functions necessary to
start the Healthy Walk program. It asks the users
what modes they want (terminal or GUI). They also can
reload the json with the nodes corresponding to tags.
"""
import subprocess
from environment import *
import stable_baselines3 as sb3
import matplotlib.pyplot as plt
import threading
from utils import LogReward, display_learning_curve, DisplayType


def get_nodes_by_tags(location):
    """
    Function that retrieves the nodes using different tags. The idea is to have a location and use OSMNx to
    retrieve the nodes containing the tags. They are in format GeoDataFrame. As explained in the report,
    we are interested in 5 group of nodes that represents different element related to a walk, based on the work
    Saelens, B. E., & Handy, S. L. (2008). Built environment correlates of walking: A review. Medicine & Science
    in Sports & Exercise, 40 (7), S550–S566. https://doi.org/10.1249/MSS.0b013e31817c67a4. The tags were found
    using the OpenStreetMap wiki : https://wiki.openstreetmap.org/wiki/Map_featureshttps://wiki.openstreetmap.org/wiki/Map_features
    """
    nature = ox.features.features_from_place(location, tags = {"leisure":
        ["fishing", "garden", "marina", "nature_reserve", "park", "picnic_table", "playground", "pitch", "slipway",
        "swimming_area", "beach_resort"], "natural" : ["heath", "scrub", "wood", "tundra", "bay", "beach", "coastline",
                                                       "water", "cliff", "dune", "hill", "valley"]} )
    must_been_places = ox.features.features_from_place(location, tags={"tourism": True, "historic": True}) # source : https://pythongis.org/part2/chapter-09/nb/00-retrieving-osm-data.html
    amenity = ox.features.features_from_place(location, tags={"amenity": True})  # amenity = places made for pedestrian
    dangerous_ways_obstacle = ox.features.features_from_place(location, tags={
        "highway": ["motorway", "trunk", "primary", "secondary", "steps"],
        "maxspeed": ["50", "60", "70", "80", "90", "100", "110", "120", "130", "140"],
        "landuse": ["cemetery", "military"],
        "smoothness": ["bad", "very_bad", "horrible", "very_horrible", "impassable"],
        "tracktype": ["grade3", "grade4", "grade5"], "surface": ["grass", "dirt", "earth", "mud", "sand", "ground"]})
    pedestrian = ox.features.features_from_place(location, tags={"highway": ["pedestrian", "living_street"],
                                                                      "footway": ["crossing", "sidewalk"],
                                                                      "maxspeed": ["20", "30", "40"]})

    graphOfWalks = ox.graph.graph_from_place(location, network_type="walk")
    nodes = ox.convert.graph_to_gdfs(graphOfWalks, edges=False)  # source : https://www.timlrx.com/blog/cleaning-openstreetmap-intersections-in-python/
    nodes_nature = gpd.sjoin(nodes, nature, how="inner", predicate="intersects") # source : https://www.timlrx.com/blog/cleaning-openstreetmap-intersections-in-python/
    nodes_must_been_places = gpd.sjoin(nodes, must_been_places, how="inner", predicate="intersects")
    nodes_amenity = gpd.sjoin(nodes, amenity, how="inner", predicate="intersects")
    nodes_dangerous_scary = gpd.sjoin(nodes, dangerous_ways_obstacle, how="inner", predicate="intersects")
    nodes_pedestrian = gpd.sjoin(nodes, pedestrian, how="inner", predicate="intersects")

    return set(nodes_nature.index), set(nodes_must_been_places.index), set(nodes_amenity.index), set(
        nodes_dangerous_scary.index), set(nodes_pedestrian.index)


def reload_json_location():
    """
    Function that retrieves the nodes in Ulqin of the 5 catgories and put the node id in a json organized by
    categories. The code is inspired by :  https://realpython.com/python-json/
    """
    natural_places, must_been_places, amenity_places, must_avoid_places, pedestrian_places = get_nodes_by_tags("Ulcinj, Montenegro")
    data = {
        "natural": list(natural_places),
        "must_been": list(must_been_places),
        "amenity": list(amenity_places),
        "must_avoid": list(must_avoid_places),
        "pedestrian": list(pedestrian_places),
    }
    with open("tags_locations.json", "w") as f:
        json.dump(data, f)


def start_tensorboard():
    """
    Function that starts the tensorboard sever for GUI. The idea is to allow
    analysing the agent behaviour. We used that as it was talked during classes,
    and we found that interesting for visualizing other metrics.
    """
    subprocess.run(["tensorboard", "--logdir", "./tensorboard/"])


def run_environment(gui_mode):
    """
    Function that runs the environment depending on the display option selected.
    It either starts the GUI or the terminal version. The GY also starts the tensorserver
    """
    if gui_mode == DisplayType.NO_DISPLAY:
        env = HealthyWalkEnv()
        agent = sb3.PPO(
            "MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log="./tensorboard/"
        )
        log_reward = LogReward()
        agent.learn(total_timesteps=40000, log_interval=1, callback=log_reward) # source : https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
        fig = display_learning_curve(log_reward)
        fig.savefig("learning_curve.png")
        plt.show()
    else:
        t = threading.Thread(target=start_tensorboard) # two thread to avoid problem of blocking, like running first command block the second will also block
        t.start()
        subprocess.run(["streamlit", "run", "gui.py", "--"]) # source : https://www.geeksforgeeks.org/python/executing-shell-commands-with-python/

def main():
    """
    Main function that starts the program. The idea is to first ask the users what they want.
    They can either run the terminal mode (No Graphical User Interface) or the GUI (Graphical
    User Interface) mode. They also have the opportunity to reload dynamically the tags if there
    are changes or some problem occurring.
    """
    user_input = None
    gui_mode = DisplayType.NO_DISPLAY
    while True: # check to be sure user doesn't put something else than the 3 options
        user_input = input( "Welcome. What do you want : \n[1] No Graphical User Interface \n"
            "[2] Graphical User Interface\n[3] Reload the json with tags \nPlease write 1 or 2 or 3 : "
        )
        if user_input == "1":
            gui_mode = DisplayType.NO_DISPLAY
            break
        elif user_input == "2":
            gui_mode = DisplayType.SIMPLE_DISPLAY
            break
        elif user_input == "3":
            print("Currently reloading the json. It may take some time... \n")
            reload_json_location()
            print("Json successfully reloaded \n")
        else:
            print("Please enter a valid input.")

    run_environment(gui_mode)

main()