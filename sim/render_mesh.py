import os
import glob
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

import numpy as np
import mujoco
import subprocess
import subprocess
from transforms3d import euler

from assets.icon_process import extract_contours

color_map = np.asarray([
    [0, 0, 0],
    [255, 255, 255],
], dtype=np.uint8)
color_maps = np.concatenate([color_map for _ in range(32)], axis=0)

def render_mesh(gripper_root: str):
    subprocess.call(['cp', 'assets/gripper_render.xml', os.path.join(gripper_root, 'gripper_render.xml')])
    model = mujoco.MjModel.from_xml_path(os.path.join(gripper_root, 'gripper_render.xml'))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 256, 256)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 0.0]
    camera.distance = 0.9
    camera.azimuth = 180
    camera.elevation = -30

    mujoco.mj_step(model, data)
    renderer.update_scene(data, camera)
    img = renderer.render()
    return img

def render_object_mesh(object_root, z_rots):
    subprocess.call(['cp', 'assets/object_render.xml', os.path.join(object_root, 'object_render.xml')])
    model = mujoco.MjModel.from_xml_path(os.path.join(object_root, 'object_render.xml'))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 128, 128)
    renderer.enable_segmentation_rendering()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 0.0]
    camera.distance = 0.8
    camera.azimuth = 135
    camera.elevation = -45
    obj_root_idx = [model.joint(jointid).name for jointid in range(model.njnt)].index("object_root")
    obj_jnt = model.joint(obj_root_idx)
    assert obj_jnt.type == 0  # freejoint
    contours = []
    for z_rot in z_rots:
        data.qpos[obj_jnt.qposadr[0] : obj_jnt.qposadr[0] + 3] = [0, 0, 0,]
        data.qpos[
            obj_jnt.qposadr[0] + 3 : obj_jnt.qposadr[0] + 7
        ] = euler.euler2quat(0, 0, z_rot)
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera)
        img = renderer.render()[..., 0]
        img = color_maps[img]
        contour = extract_contours(img, num_points=100, rescale=False)
        contours.append(contour)
    return contours