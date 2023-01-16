# wrapper functions for object centric environments to
# ensure they publish their state as a dict of GameObjects

from scobi.utils.game_object import GameObject
oc_game_objects = {}

# OC_Atari step wrapper
def ocatari_step(ocatari_step_return):
    obs_raw, reward, truncated, terminated, info = ocatari_step_return
    gameobject_info = info["objects"]
    for key in gameobject_info:
        try:
            oc_game_objects[key].rgb = [gameobject_info[key][4], gameobject_info[key][5], gameobject_info[key][6]]
            oc_game_objects[key].update_coords(gameobject_info[key][0],gameobject_info[key][1])
        except KeyError:
            tmp = gameobject_info[key]
            rgb = [tmp[4], tmp[5], tmp[6]]
            wh = [tmp[2], tmp[3]]
            oc_game_objects[key] = GameObject(key, rgb, wh)
    return oc_game_objects, reward, truncated, terminated, info, obs_raw