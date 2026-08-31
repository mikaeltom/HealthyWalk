"""
Mikael Tom
9 June 2026
File that contains what is necessary for
running the streamlit GUI in order to display the
training and learning curves.
"""
import streamlit as st
import streamlit.components.v1 as components
import threading
from environment import *
import time
from utils import LogReward, display_learning_curve, DisplayType
from streamlit.runtime.scriptrunner import add_script_run_ctx
import matplotlib.pyplot as plt


# The following portions of code contain all what it is necessary for running
# the GUI. We used streamlit for that as it is makes light and easy to use
# interfaces.
# All the following portion of codes were inspired by the documentation :
# https://docs.streamlit.io/develop/api-reference/layout/st.container
# https://docs.streamlit.io/develop/api-reference/layout/st.columns
# https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state

if "display_mode" not in st.session_state:
    st.session_state.display_mode = DisplayType.SIMPLE_DISPLAY # streamlit put variables in st.session_state

def start_training():
    """
    Function that creates the environment and starts the training for the agent on a different thread.
    It was necessary to run it on a separate thread as the GUI itself need a thread.
    """
    st.session_state.started_training = True
    display_mode = st.session_state.display_mode
    env = HealthyWalkEnv(display=display_mode)
    st.session_state.env = env
    training_thread = threading.Thread(target=run_training, args=(env,))
    add_script_run_ctx(training_thread) # source : https://discuss.streamlit.io/t/warning-for-missing-scriptruncontext/83893
    training_thread.start()

def run_training(env):
    """
    Function used by the thread to run the agent training
    """
    agent = sb3.PPO(
        "MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )
    log_reward = LogReward()
    agent.learn(total_timesteps=40000, log_interval=1, tb_log_name="run", callback=log_reward)
    st.session_state.log_reward = log_reward
    st.session_state.started_training = False # reset the booleans
    st.session_state.training_complete = True

st.set_page_config(layout="wide")
if "started_training" not in st.session_state:
    st.session_state.started_training = False

st.title("_HealthyWalk_", text_alignment="center")
st.divider()

if "training_complete" not in st.session_state:
    st.session_state.training_complete = False

if "plot_rewards" not in st.session_state:
    st.session_state.plot_rewards = False

if st.session_state.training_complete:
    st.toast("Training is done. You can retry if wanted") # source : https://docs.streamlit.io/develop/api-reference/status/st.toast
    st.session_state.training_complete = False
    st.session_state.plot_rewards = True

_, center, _  = st.columns(3) # button for the training start
with center:
    if not st.session_state.started_training:
        st.button("Press to start training", on_click=start_training, use_container_width=True) # source : https://docs.streamlit.io/develop/api-reference/widgets/st.button
    else:
        st.success("Training just started. Have a look to the training")

_, center_toggle, _ = st.columns([3,1,3]) # toogle for saying if we want the user feedback mode or not
with center_toggle:
    if not st.session_state.started_training:
        if st.toggle("Activate reward feedback"): # source : https://docs.streamlit.io/develop/api-reference/widgets/st.toggle
            st.session_state.display_mode = DisplayType.RATING_DISPLAY
        else:
            st.session_state.display_mode = DisplayType.SIMPLE_DISPLAY

st.space("small")

col1, col2= st.columns(2) # used for displaying the tensorfboard using iframe feature from streamlit
with col1:
    st.text("Tensorboard statistics : ")
    iframe_src = "http://localhost:6006/" # source : https://discuss.streamlit.io/t/how-do-i-embed-an-existing-non-streamlit-webpage-to-my-streamlit-app/50326/3
    components.iframe(iframe_src, height=700, width=1200)

@st.fragment(run_every=1) # source : https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment
def update_map():
    """
    Function that handle the updating of the map. The idea is to retrieve from the
    environment informations about the map of the location, the different nodes that
    will be colored with their category, the agent end episode path and the requested
    time and loop time. Using that it displays all this using streamlit. It also
    handles the display of the 5 stars to click on for rating (if mode selected by user).
    """
    if st.session_state.training_complete: # reload if training done
        st.rerun()

    env = st.session_state.env
    if env.unwrapped.current_map_to_display is not None:
        natural = env.unwrapped.natural_places
        must_been = env.unwrapped.must_been_places
        amenity = env.unwrapped.amenity_places
        must_avoid = env.unwrapped.must_avoid_places
        pedestrian = env.unwrapped.pedestrian_places
        node_colors = [
            '#5DE488' if node in natural
            else '#FFFFC1' if node in must_been
            else '#9B9C9E' if node in amenity
            else '#EF6667' if node in must_avoid
            else '#FFBD44' if node in pedestrian
            else '#FFFFFF'
            for node in env.unwrapped.graphOfWalks.nodes()
        ]
        fig, ax = ox.plot_graph_route(env.unwrapped.graphOfWalks, env.unwrapped.current_map_to_display,
                node_color=node_colors, route_color="#3C9DF3", show=False) # source : https://www.geeksforgeeks.org/python/find-shortest-path-using-python-osmnx-routing-module/
        map_place_holder.pyplot(fig) # source : https://docs.streamlit.io/develop/api-reference/charts/st.pyplot
        st.markdown("**Key** : :green[Natural places]   :yellow[Must been places]   :grey[Amenity places]   :red[Must avoid places]   :orange[Pedestrian places]   :blue[Current loop]")
        time_required, current_time = env.unwrapped.current_time_to_display
        st.markdown(f"Requested time : {time_required} - Current time : {current_time}")
        plt.close(fig)

    if env.unwrapped.display_behaviour == DisplayType.RATING_DISPLAY.value and env.unwrapped.asking_rating:
        st.text("Give a rating for this walk:")
        feedback_mapping = [1, 2, 3, 4, 5]
        selected = st.feedback("stars", key="episode_feedback")  # source : https://docs.streamlit.io/develop/api-reference/widgets/st.feedback
        if selected is not None:
            rating = feedback_mapping[selected]
            env.unwrapped.received_rating = rating
            env.unwrapped.asking_rating = False
            st.success("Thanks for the feeback : " + str(rating) + "/5")
            time.sleep(0.5) # to avoid that the confirmation message is removed to fast

with col2: # used for displaying the updated map
    st.text("Walk suggested by a training episode :")
    map_place_holder = st.empty()
    if st.session_state.started_training:
        update_map()
    else: # if didn't start we just put the empty map
        g = ox.graph_from_place("Ulcinj, Montenegro", network_type="walk")
        fig, ax = ox.plot_graph(g, show=False, close=True)
        map_place_holder.pyplot(fig) # source : https://discuss.streamlit.io/t/how-to-display-matplotlib-graphs-in-streamlit-application/35383/2"""

st.space("small")
st.text("Learning curve produced by training :")
_, mid_bottom, _ = st.columns([1, 2, 1])
with mid_bottom: # used for displaying the learning curve
    if st.session_state.plot_rewards:
        if st.session_state.log_reward:
            fig = display_learning_curve(st.session_state.log_reward)
            st.pyplot(fig)
        st.session_state.plot_rewards = False
    else:
        st.info("Start the training to see this graph")
