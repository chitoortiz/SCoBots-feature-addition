"""scobi core"""
import numpy as np
from gymnasium import spaces, Env
import scobi.environments.env_manager as em
from scobi.utils.game_object import get_wrapper_class
from scobi.focus import Focus
from scobi.utils.logging import Logger
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy

class Environment(Env):
    def __init__(
            self,
            env_name,
            seed=None,
            focus_dir="resources/focusfiles",
            focus_file=None,
            reward=0,
            hide_properties=False,
            silent=False,
            refresh_yaml=True,
            draw_features=False,
            hud=False
    ):
        self.logger = Logger(silent=silent)
        # Create the underlying Atari (or other) environment
        self.oc_env = em.make(env_name, self.logger, hud=hud, buffer_window_size=2)
        self.seed = seed
        self.randomstate = np.random.RandomState(self.seed)
        self.game_object_wrapper = get_wrapper_class()

        self.oc_env.reset(seed=self.seed)

        # Some other scobi logic...
        self.noisy_objects = False
        init_objects = self.oc_env._slots
        max_obj_dict = self.oc_env.max_objects_per_cat

        self.did_reset = False
        self.focus = Focus(
            env_name,
            reward,
            hide_properties,
            focus_dir,
            focus_file,
            init_objects,
            max_obj_dict,
            self.oc_env._env.unwrapped.get_action_meanings(),
            refresh_yaml,
            self.logger
        )
        self.focus_file = self.focus.FOCUSFILEPATH
        self.action_space = spaces.Discrete(len(self.focus.PARSED_ACTIONS))
        self.action_space_description = self.focus.PARSED_ACTIONS
        self.observation_space_description = (
            self.focus.PARSED_PROPERTIES + self.focus.PARSED_FUNCTIONS
        )
        self.feature_vector_description = self.focus.get_feature_vector_description()
        self.num_envs = 1
        self.draw_features = draw_features
        self.feature_attribution = []
        self.render_font = ImageFont.truetype(
            str(Path(__file__).parent / 'resources' / 'Gidole-Regular.ttf'), size=38
        )
        self.obj_obs = None
        self._rel_obs = None
        self._top_features = []

        self.original_obs = []
        self.original_reward = []
        self.ep_env_reward = None
        self.ep_env_reward_buffer = 0
        self.reset_ep_reward = True

        # Mix or partial scobi reward
        if reward == 2:  # mix
            self._reward_composition_func = lambda a, b: a + b
        elif reward == 1:  # scobi only
            self._reward_composition_func = lambda a, b: a
        else:  # env only
            self._reward_composition_func = lambda a, b: b

        # If ALE pass over
        try:
            self.focus.ale = self.oc_env._env.unwrapped.ale
        except:
            self.focus.ale = None

        self.reset()
        # Step once to set feature vector size
        self.step(0)
        self.observation_space = spaces.Box(
            low=-2**63,
            high=2**63 - 2,
            shape=(self.focus.OBSERVATION_SIZE,),
            dtype=np.float32
        )
        # Another reset
        self.reset()
        self.did_reset = False

    def step(self, action):
        if not self.did_reset:
            self.logger.GeneralError("Cannot call env.step() before calling env.reset()")
        elif self.action_space.contains(action):
            obs, base_reward, truncated, terminated, info = self.oc_env.step(action)

            sco_obs, sco_reward = self.focus.get_feature_vector(obs)
            freeze_mask = self.focus.get_current_freeze_mask()

            # Possibly draw features...
            if self.draw_features:
                img_obs = self.oc_env._state_buffer_rgb[-1]
                self.obj_obs = self._draw_objects_overlay(img_obs)
                self._rel_obs = self._draw_relation_overlay(img_obs, sco_obs, freeze_mask, action)

            self.original_obs = obs
            self.original_reward = base_reward
            self.ep_env_reward_buffer += self.original_reward
            if self.reset_ep_reward:
                self.ep_env_reward = None
                self.reset_ep_reward = False
            if terminated or truncated:
                self.ep_env_reward = self.ep_env_reward_buffer
                self.ep_env_reward_buffer = 0
                self.reset_ep_reward = True
                self.focus.reward_subgoals = 0

            final_reward = self.focus.compose_reward(sco_reward, base_reward)

            return sco_obs, final_reward, truncated, terminated, info
        else:
            raise ValueError("scobi> Action not in action space")

    def reset(self, *args, **kwargs):
        self.did_reset = True
        self.focus.reward_threshold = -1
        self.focus.reward_history = [0, 0]
        obs, info = self.oc_env.reset(*args, **kwargs)
        sco_obs, _ = self.focus.get_feature_vector(obs)
        # Reset any carry values in case of custom reward func
        self.focus.reset_carry_value()

        return sco_obs, info

    @property
    def unwrapped(self):
        return self.oc_env.unwrapped

    def close(self):
        self.oc_env.close()

    def set_feature_attribution(self, att):
        self.feature_attribution = att

    def _add_margin(self, pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def _draw_objects_overlay(self, obs_image, action=None):
        obs_mod = deepcopy(obs_image)
        for obj in self.oc_env.objects:
            mark_bb(obs_mod, obj.xywh, color=obj.rgb, name=str(obj))
        return obs_mod

    def _draw_relation_overlay(self, obs_image, feature_vector, freeze_mask, action=None):
        # ... your existing code ...
        return np.array(obs_image)


def mark_bb(image_array, bb, color=(255, 0, 0), surround=True, name=None):
    # unchanged utility
    x, y, w, h = bb
    color = _make_darker(color)
    if surround:
        if x > 0:
            x, w = bb[0] - 1, bb[2] + 1
        else:
            x, w = bb[0], bb[2]
        if y > 0:
            y, h = bb[1] - 1, bb[3] + 1
        else:
            y, h = bb[1], bb[3]
    y = min(209, y)
    x = min(159, x)
    bottom = min(209, y + h)
    right = min(159, x + w)
    image_array[y:bottom + 1, x] = color
    image_array[y:bottom + 1, right] = color
    image_array[y, x:right + 1] = color
    image_array[bottom, x:right + 1] = color

def _make_darker(color, col_precent=0.8):
    if not color:
        return [0, 0, 0]
    return [int(col * col_precent) for col in color]
    