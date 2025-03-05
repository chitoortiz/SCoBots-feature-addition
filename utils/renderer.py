import pickle
from fileinput import filename

import numpy as np
import pygame
import gymnasium as gym
import time
import os
try:
    from pygame_screen_record import ScreenRecorder
    _screen_recorder_imported = True
except ImportError as imp_err:
    _screen_recorder_imported = False

from pathlib import Path


class Renderer:
    window: pygame.Surface
    clock: pygame.time.Clock
    env: gym.Env
    zoom: int = 4
    fps: int = 20

    def __init__(self, envs, model, path, record=False, nb_frames=0, code_version=False):
        self.envs = envs
        if hasattr(envs, 'venv') and hasattr(envs.venv, 'envs'):
            self.env = envs.venv.envs[0]
            self.rgb_agent = False
        elif hasattr(envs, 'envs'):
            self.env = envs.envs[0]  # Handles cases where envs are in a DummyVecEnv or similar
            self.rgb_agent = True
        else:
            self.env = envs
            self.rgb_agent = True
        self.model = model
        self._code_version = code_version
        self.current_frame = self._get_current_frame()
        self._init_pygame(self.current_frame)
        self.paused = False

        self.path = path

        self.current_keys_down = set()
        self.current_mouse_pos = None
        if not self.rgb_agent:
            self.keys2actions = self.env.oc_env.unwrapped.get_keys_to_action()

        self.ram_grid_anchor_left = self.env_render_shape[0] + 28
        self.ram_grid_anchor_top = 28

        self.active_cell_idx = None
        self.candidate_cell_ids = []
        self.current_active_cell_input : str = ""

        self.human_playing = False
        self.print_reward = False
        self._recording = False

        if record:
            if _screen_recorder_imported:
                self._screen_recorder = ScreenRecorder(60)
                self._screen_recorder.start_rec()
                self._recording = True
                self.nb_frames = nb_frames
            else:
                print("Screen recording not available. Please install the pygame_screen_record package.")
                exit(1)
        else:
            self.nb_frames = np.inf

    def _init_pygame(self, sample_image):
        pygame.init()
        pygame.display.set_caption("OCAtari Environment")
        sample_image = np.repeat(np.repeat(np.swapaxes(sample_image, 0, 1), self.zoom, axis=0), self.zoom, axis=1)
        self.env_render_shape = sample_image.shape[:2]
        if self._code_version is not None:
            height, width = sample_image.shape[:2]
            window_size = (width + 300, self.env_render_shape[:2][1])
        else:
            window_size = self.env_render_shape[:2]
        self.window = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()      

    def run(self):
        self.running = True
        obs = self.envs.reset()
        i = 1
        while self.running:
            self._handle_user_input()
            if not self.paused:
                if self.human_playing:
                    action = [self._get_action()]
                    time.sleep(0.05)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                obs, rew, done, infos = self.envs.step(action)
                self.env.sco_obs = obs
                self.current_frame = self._get_current_frame()
                if self.print_reward and rew[0]:
                    print(rew[0])
                if done:
                    if self._recording and self.nb_frames == 0:
                        self._save_recording()
                    obs = self.envs.reset()
                elif self._recording and i == self.nb_frames:
                    self._save_recording()
            self._render()

            if self.rgb_agent:
                self.clock.tick(self.fps)
        pygame.quit()


    def run_code_version(self):
        self.running = True
        obs = self.envs.reset()
        i = 1
        while self.running:
            self._handle_user_input()
            if not self.paused:
                # Either human or AI:
                if self.human_playing:
                    action = [self._get_action()]
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                obs, rew, done, infos = self.envs.step(action)
                self.env.sco_obs = obs
                self.current_frame = self._get_current_frame()
                if done:
                    obs = self.envs.reset()
            self._render_code_version()
            i += 1
            pygame.display.flip()
            pygame.event.pump()
        pygame.quit()

    def _render_code_version(self, frame=None):
        self.window.fill((0,0,0))  # black background
        self._render_atari(frame)

        x_offset = self.env_render_shape[1] + 10

        y_offset = 10

        # write headline and centering
        text_region_width = 300
        text_area_center_x = x_offset + text_region_width // 6

        headline = "Decision Path if-Checks"
        headline_font = pygame.font.SysFont(None, 50)
        headline_color = (255, 255, 0)
        headline_surface = headline_font.render(headline, True, headline_color)

        # Center the headline
        headline_rect = headline_surface.get_rect()
        # top is y_offset
        headline_rect.top = y_offset
        headline_rect.centerx = text_area_center_x

        self.window.blit(headline_surface, headline_rect)

        y_offset = 70

        # write each line of path
        if hasattr(self.model, 'decision_path'):
            line_height = 50
            for line in self.model.decision_path:
                self._draw_text(line, self.env_render_shape[0] + 10, y_offset, font_size=37)
                y_offset += line_height


    def _draw_text(self, text, x, y, color=(255,255,255), font_size=18):
        font = pygame.font.SysFont(None, font_size)
        text_surface = font.render(text, True, color)
        self.window.blit(text_surface, (x, y))

    def _get_action(self):
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            return self.keys2actions[pressed_keys]
        else:
            return 0  # NOOP

    def _save_recording(self):
        filename = Path.joinpath(self.path, "recordings")
        filename.mkdir(parents=True, exist_ok=True)
        self._screen_recorder.stop_rec() # stop recording
        print(self.env.spec.name)
        if self.rgb_agent:
            filename = Path.joinpath(filename, f"{self.env.spec.name}.avi")
        else:
            filename = Path.joinpath(filename, f"{self.env.oc_env.game_name}.avi")
        i = 0
        while os.path.exists(filename):
            i += 1
            if self.rgb_agent:
                filename = Path.joinpath(self.path, "recordings", f"{self.env.spec.name}_{i}.avi")
            else:
                filename = Path.joinpath(self.path, "recordings", f"{self.env.oc_env.game_name}_{i}.avi")
        print(filename)
        self._screen_recorder.save_recording(filename)
        print(f"Recording saved as {filename}")
        self._recording = False

    def _handle_user_input(self):
        self.current_mouse_pos = np.asarray(pygame.mouse.get_pos())

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                if self._recording:
                    self._save_recording()
                self.running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_p:  # 'P': pause/resume
                    self.paused = not self.paused

                elif event.key == pygame.K_r:  # 'R': reset
                    self.env.reset()
                
                elif event.key == pygame.K_h:  # 'H': toggle human playing
                    self.human_playing = not self.human_playing
                    if self.human_playing:
                        print("Human playing")
                    else:
                        print("AI playing")
                
                elif event.key == pygame.K_m:  # 'M': save snapshot
                    snapshot = self.env._ale.cloneState()
                    pickle.dump(snapshot, open("snapshot.pkl", "wb"))
                    print("Saved snapshot.pkl")

                elif event.key == pygame.K_ESCAPE and self.active_cell_idx is not None:
                    self._unselect_active_cell()

                elif (event.key,) in self.keys2actions.keys():  # env action
                    self.current_keys_down.add(event.key)

                elif pygame.K_0 <= event.key <= pygame.K_9:  # enter digit
                    char = str(event.key - pygame.K_0)
                    if self.active_cell_idx is not None:
                        self.current_active_cell_input += char

                elif event.key == pygame.K_BACKSPACE:  # remove character
                    if self.active_cell_idx is not None:
                        self.current_active_cell_input = self.current_active_cell_input[:-1]

                elif event.key == pygame.K_RETURN:
                    if self.active_cell_idx is not None:
                        if len(self.current_active_cell_input) > 0:
                            new_cell_value = int(self.current_active_cell_input)
                            if new_cell_value < 256:
                                self._set_ram_value_at(self.active_cell_idx, new_cell_value)
                        self._unselect_active_cell()

            elif event.type == pygame.KEYUP:  # keyboard key released
                if (event.key,) in self.keys2actions.keys():
                    self.current_keys_down.remove(event.key)

    def _get_current_frame(self):
        if self.rgb_agent:
            return self.env.render()
        else:
            return self.env.obj_obs

    def _render(self, frame = None):
        self.window.fill((0,0,0))  # clear the entire window
        self._render_atari(frame)
        pygame.display.flip()
        pygame.event.pump()

    def _render_atari(self, frame = None):
        if frame is None:
            frame = self.current_frame
        frame = np.swapaxes(np.repeat(np.repeat(frame, self.zoom, axis=0), self.zoom, axis=1), 0, 1)
        frame_surface = pygame.Surface(self.env_render_shape)
        pygame.pixelcopy.array_to_surface(frame_surface, frame)
        self.window.blit(frame_surface, (0, 0))
        self.clock.tick(60)
