import os
import glob
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
from typing import Optional, final

import mujoco
from transforms3d import euler, quaternions
import numpy as np
import subprocess
from mujoco import viewer
import ray
import subprocess
import time
import imageio
import cv2

from dynamics.utils import continuous_signed_delta, visualize_profile, visualize_finals
from sim.sim_3d import prepare_object
from assets.finger_3d import save_3d_gripper, generate_gripper_3d_xml, generate_scene_3d_xml
from sim.render_mesh import render_mesh, render_object_mesh

threshold = np.array([0.02, 0.001, 0.001])

def compute_collision(mesh_path, num_retries: int = 2):
    """
    Computes the convex decomposition of a mesh using v-hacd.
    Convention: the input mesh is assumed to be in the same folder as the output mesh,
    with only the name change from `xyz.obj` to `xyz_collision.obj`.

    V-HACD help:
    ```
    -h <n>                  : Maximum number of output convex hulls. Default is 32
    -r <voxelresolution>    : Total number of voxels to use. Default is 100,000
    -e <volumeErrorPercent> : Volume error allowed as a percentage. Default is 1%. Valid range is 0.001 to 10
    -d <maxRecursionDepth>  : Maximum recursion depth. Default value is 10.
    -s <true/false>         : Whether or not to shrinkwrap output to source mesh. Default is true.
    -f <fillMode>           : Fill mode. Default is 'flood', also 'surface' and 'raycast' are valid.
    -v <maxHullVertCount>   : Maximum number of vertices in the output convex hull. Default value is 64
    -a <true/false>         : Whether or not to run asynchronously. Default is 'true'
    -l <minEdgeLength>      : Minimum size of a voxel edge. Default value is 2 voxels.
    -p <true/false>         : If false, splits hulls in the middle. If true, tries to find optimal split plane location. False by default.
    -o <obj/stl/usda>       : Export the convex hulls as a series of wavefront OBJ files, STL files, or a single USDA.
    -g <true/false>         : If set to false, no logging will be displayed.
    ```
    """
    COMMAND = [
        "TestVHACD",
        mesh_path,
        "-r",
        "100000",
        "-o",
        "obj",
        "-g",
        "false",
        "-h",
        "32",
        "-v",
        "32",
    ]
    output: Optional[subprocess.CompletedProcess] = None
    assert num_retries > 1
    for _ in range(num_retries):
        try:
            output = subprocess.run(COMMAND, check=True)
        except subprocess.CalledProcessError as e:
            print("V-HACD failed to run on %s, retrying..." % mesh_path)
            continue
    if output is None or output.returncode != 0:
        raise RuntimeError("V-HACD failed to run on %s" % mesh_path)

def prepare_gripper(gripper_idx: int, ctrlpts, model_root: str):
    save_gripper_dir = os.path.join(model_root, 'grippers', str(gripper_idx))
    if os.path.exists(save_gripper_dir):
        return save_gripper_dir
    else:
        save_3d_gripper(
            ctrlpts[:len(ctrlpts) // 2],
            ctrlpts[len(ctrlpts) // 2:],
            width=0.1,
            sample_size=25,
            save_gripper_dir=save_gripper_dir,
        )
        meshl_path = os.path.join(save_gripper_dir, "fingerl.obj")
        compute_collision(meshl_path)
        meshr_path = os.path.join(save_gripper_dir, "fingerr.obj")
        compute_collision(meshr_path)
        generate_gripper_3d_xml(len(glob.glob(os.path.join(save_gripper_dir, "fingerl0*.obj"))), len(glob.glob(os.path.join(save_gripper_dir, "fingerr0*.obj"))), gripper_idx, os.path.join(model_root, 'gripper_%d.xml' % gripper_idx))
    return save_gripper_dir

@ray.remote(num_cpus=2)
def sim_test(ctrlpts, object_name: str, gripper_idx: int=0, object_idx: int=0, object_order_idx: int=0, model_root: str="assets", save_dir: str="sim", gui: bool = False, render: bool = True, num_rot: int = 360, ori_range: list = [-1.0, 1.0], render_last: bool = False):
    save_gripper_dir = prepare_gripper(gripper_idx, ctrlpts, model_root)
    while not (os.path.exists(os.path.join(model_root, 'gripper_%d.xml' % gripper_idx)) and os.path.getsize(os.path.join(model_root, 'gripper_%d.xml' % gripper_idx))>0):
        time.sleep(0.1)
    gripper_img = render_mesh(save_gripper_dir)
    gripper_img_path = os.path.join(save_dir, '%d_%d_gripper.png' % (object_idx, gripper_idx))
    cv2.imwrite(gripper_img_path, gripper_img)

    save_object_dir = prepare_object(object_name, object_idx, model_root)
    while not (os.path.exists(os.path.join(model_root, 'object_%d.xml' % object_idx)) and os.path.getsize(os.path.join(model_root, 'object_%d.xml' % object_idx))>0):
        time.sleep(0.1)
    contours = render_object_mesh(save_object_dir, np.linspace(ori_range[0], ori_range[1], num_rot//36) * np.pi + np.pi)

    scene_path = os.path.join(model_root, 'scene_%d_%d.xml' % (object_idx, gripper_idx))
    generate_scene_3d_xml(object_idx, gripper_idx, scene_path)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    reset_qpos = data.qpos.copy()
    reset_qvel = data.qvel.copy()
    reset_force = data.qfrc_applied.copy()
    handle = viewer.launch_passive(model, data) if gui else None
    obj_root_idx = [model.joint(jointid).name for jointid in range(model.njnt)].index("object_root")
    obj_jnt = model.joint(obj_root_idx)
    assert obj_jnt.type == 0  # freejoint

    left_grip_idx = [model.joint(jointid).name for jointid in range(model.njnt)].index("left_grip")
    left_grip_jnt = model.joint(left_grip_idx)
    right_grip_idx = [model.joint(jointid).name for jointid in range(model.njnt)].index("right_grip")
    right_grip_jnt = model.joint(right_grip_idx)

    if render or render_last:
        renderer = mujoco.Renderer(model, 128, 128)
        # renderer.enable_segmentation_rendering()
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, 0.0, 0.0]
        camera.distance = 0.8
        camera.azimuth = 135
        camera.elevation = -45

    z_rots = np.linspace(ori_range[0], ori_range[1], num_rot) * np.pi + np.pi
    init_poses = np.zeros((len(z_rots), 7))
    final_poses = np.zeros((len(z_rots), 7))
    imgs = np.zeros((len(z_rots) // 36, 800, 128, 128, 3), dtype=np.int8)
    # segs = np.zeros((len(z_rots) // 36, 100, 128, 128), dtype=np.int16)
    final_final_poses = np.zeros((len(z_rots), 7))
    for k, z_rot in enumerate(z_rots):
        data.qpos[:] = reset_qpos[:].copy()
        data.qvel[:] = reset_qvel[:].copy()
        data.qfrc_applied[:] = reset_force
        data.qpos[obj_jnt.qposadr[0] : obj_jnt.qposadr[0] + 3] = [0, 0, 0,]
        data.qpos[
            obj_jnt.qposadr[0] + 3 : obj_jnt.qposadr[0] + 7
        ] = euler.euler2quat(0, 0, z_rot)
        init_poses[k, :] = data.qpos[
            obj_jnt.qposadr[0] : obj_jnt.qposadr[0] + 7
        ]
        data.ctrl[0] = 0.5
        data.ctrl[1] = -0.5
        for t in range(32000):
            if handle is not None and t % 10 == 0:
                handle.sync()
                input(f"Press Enter to continue..., {t}")
            if t % 800 == 0 and t > 0:
                # reset the positions velocities forces of the gripper
                data.qpos[left_grip_jnt.qposadr[0]] = reset_qpos[left_grip_jnt.qposadr[0]]
                data.qpos[right_grip_jnt.qposadr[0]] = reset_qpos[right_grip_jnt.qposadr[0]]
                data.qvel[:] = reset_qvel[:]
                data.qfrc_applied[:] = reset_force[:]
            mujoco.mj_step(model, data)
            if (render and k % 36 == 0 and t % 40 == 0) or (render_last and  k % 36 ==0 and t == 7999):
                renderer.update_scene(data, camera)
                img = renderer.render()
                # seg = renderer.render()[..., 0]
                # segs[k // 36, t // 40, ...] = seg
                # img = color_maps[seg]
                imgs[k // 36, t // 40, ...] = img
            if t == 800:
                final_poses[k, :] = data.qpos[
                    obj_jnt.qposadr[0] : obj_jnt.qposadr[0] + 7
                ]
            final_final_poses[k, :] = data.qpos[
                    obj_jnt.qposadr[0] : obj_jnt.qposadr[0] + 7
            ]

    save_data = {
        "obj_pos": init_poses[..., :3].reshape((-1, 3)),
        "obj_theta": np.asarray([quaternions.quat2axangle(quat)[-1] for quat in init_poses[..., 3:].reshape((-1, 4))], dtype=np.float32),
        "delta_theta": np.asarray([continuous_signed_delta(quaternions.quat2axangle(last_quat)[-1], quaternions.quat2axangle(quat)[-1]) for last_quat, quat in zip(init_poses[..., 3:].reshape((-1, 4)), final_poses[..., 3:].reshape((-1, 4)))], dtype=np.float32),    # shape: (num_rot,)
        "delta_pos": (final_poses[..., :3] - init_poses[..., :3]).reshape((-1, 3)),
    }
    os.makedirs(os.path.join(save_dir, '%d_%d' % (object_idx, gripper_idx)), exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, "%d_%d.npz" % (object_idx, gripper_idx)), save_data)
    # visualize and save profile
    profile = np.asarray([1 if delta_theta > threshold[0] else -1 if delta_theta < -threshold[0] else 0 for delta_theta in save_data['delta_theta']])
    profile_x = np.asarray([1 if delta_pos[0] > threshold[1] else -1 if delta_pos[0] < -threshold[1] else 0 for delta_pos in save_data['delta_pos']])
    profile_y = np.asarray([1 if delta_pos[1] > threshold[2] else -1 if delta_pos[1] < -threshold[2] else 0 for delta_pos in save_data['delta_pos']])
    visualize_profile(profile, os.path.join(save_dir, '%d_%d_profile.png' % (object_idx, gripper_idx)), ori_range=ori_range)
    visualize_profile(profile_x, os.path.join(save_dir, '%d_%d_profile_x.png' % (object_idx, gripper_idx)), ori_range=ori_range)
    visualize_profile(profile_y, os.path.join(save_dir, '%d_%d_profile_y.png' % (object_idx, gripper_idx)), ori_range=ori_range)
    final_thetas = np.asarray([quaternions.quat2axangle(quat)[-1] for quat in final_final_poses[:, 3:].reshape((-1, 4))], dtype=np.float32)
    final_delta_thetas = np.asarray([continuous_signed_delta(init_theta, final_theta) for final_theta, init_theta in zip(final_thetas, save_data['obj_theta'])], dtype=np.float32)
    visualize_finals(final_thetas, os.path.join(save_dir, '%d_%d_final.png' % (object_idx, gripper_idx)))
    metrics = {
        'delta_theta': save_data['delta_theta']*180/np.pi,
        'delta_pos': save_data['delta_pos']*100,
        'profile': profile + 1,
        'profile_x': profile_x + 1,
        'profile_y': profile_y + 1,
        'final_theta': final_thetas*180/np.pi,
        'final_delta_theta': final_delta_thetas*180/np.pi,
        'final_pos': final_final_poses[:, :3]*100,
    }

    if render:
        videos = []
        for video_idx, video in enumerate(imgs):
            with imageio.get_writer(os.path.join(save_dir, '%d_%d' % (object_idx, gripper_idx), '%d.mp4' % video_idx), fps=20) as writer:
                for frame_idx, frame in enumerate(video):
                    cv2.drawContours(frame, [contours[video_idx]], -1, (38, 80, 115), 1)
                    writer.append_data(frame.astype(np.uint8))
            videos.append(os.path.join(save_dir, '%d_%d' % (object_idx, gripper_idx), '%d.mp4' % video_idx))
        return gripper_img_path, metrics, os.path.join(save_dir, '%d_%d_profile.png' % (object_idx, gripper_idx)), os.path.join(save_dir, '%d_%d_profile_x.png' % (object_idx, gripper_idx)), os.path.join(save_dir, '%d_%d_profile_y.png' % (object_idx, gripper_idx)), os.path.join(save_dir, '%d_%d_final.png' % (object_idx, gripper_idx)), videos, gripper_idx, object_order_idx, save_gripper_dir
    elif render_last:
        last_imgs = []
        for video_idx, video in enumerate(imgs):
            img_last = video[-1].copy()
            cv2.drawContours(img_last, [contours[video_idx]], -1, (38, 80, 115), 1)
            cv2.imwrite(os.path.join(save_dir, '%d_%d' % (object_idx, gripper_idx), '%d.png' % video_idx), img_last)
            last_imgs.append(os.path.join(save_dir, '%d_%d' % (object_idx, gripper_idx), '%d.png' % video_idx))
        return gripper_img_path, metrics, os.path.join(save_dir, '%d_%d_profile.png' % (object_idx, gripper_idx)), os.path.join(save_dir, '%d_%d_profile_x.png' % (object_idx, gripper_idx)), os.path.join(save_dir, '%d_%d_profile_y.png' % (object_idx, gripper_idx)), os.path.join(save_dir, '%d_%d_final.png' % (object_idx, gripper_idx)), last_imgs, gripper_idx, object_order_idx, save_gripper_dir
    else:
        return gripper_img_path, metrics, os.path.join(save_dir, '%d_%d_profile.png' % (object_idx, gripper_idx)), os.path.join(save_dir, '%d_%d_profile_x.png' % (object_idx, gripper_idx)), os.path.join(save_dir, '%d_%d_profile_y.png' % (object_idx, gripper_idx)), os.path.join(save_dir, '%d_%d_final.png' % (object_idx, gripper_idx)), gripper_idx, object_order_idx, save_gripper_dir

def sim_test_batch_3d(ctrlpts_y, object_names, save_dir, num_cpus=32, num_rot=360, ori_range=[-1.0, 1.0], render=True, render_last=False):
    model_root = os.path.join(save_dir, 'sim_model')
    num_gripper = ctrlpts_y.shape[0]
    ray.init(num_cpus=num_cpus, log_to_driver=False)
    ray_tasks = []
    for i, object_name in enumerate(object_names):
        for idx, p_y in enumerate(ctrlpts_y):
            p_y = p_y.reshape(-1)
            p_y = p_y * 0.05 - 0.05     # scale p_y from [-1, 1] to [-0.1, 0]
            ray_tasks.append(sim_test.remote(ctrlpts=p_y, object_name=object_name, gripper_idx=idx, object_idx=i, object_order_idx=i, model_root=model_root, save_dir=save_dir, gui=False, render=render, num_rot=num_rot, ori_range=ori_range, render_last=render_last))
    gripper_imgs, metrics, profiles, profiles_x, profiles_y, finals, videos, save_gripper_dirs = {}, {}, {}, {}, {}, {}, {}, {}
    while len(ray_tasks) > 0:
        ready, ray_tasks = ray.wait(ray_tasks, num_returns=1)
        try:
            if render or render_last:
                gripper_img_path, metric, profile, profile_x, profile_y, final, video, gripper_idx, object_idx, save_gripper_dir = ray.get(ready[0])
                gripper_imgs[object_idx * num_gripper + gripper_idx] = gripper_img_path
                metrics[object_idx * num_gripper + gripper_idx] = metric
                profiles[object_idx * num_gripper + gripper_idx] = profile
                profiles_x[object_idx * num_gripper + gripper_idx] = profile_x
                profiles_y[object_idx * num_gripper + gripper_idx] = profile_y
                finals[object_idx * num_gripper + gripper_idx] = final
                videos[object_idx * num_gripper + gripper_idx] = video
                save_gripper_dirs[object_idx * num_gripper + gripper_idx] = save_gripper_dir
            else:
                gripper_img_path, metric, profile, profile_x, profile_y, final, gripper_idx, object_idx, save_gripper_dir = ray.get(ready[0])
                gripper_imgs[object_idx * num_gripper + gripper_idx] = gripper_img_path
                metrics[object_idx * num_gripper + gripper_idx] = metric
                profiles[object_idx * num_gripper + gripper_idx] = profile
                profiles_x[object_idx * num_gripper + gripper_idx] = profile_x
                profiles_y[object_idx * num_gripper + gripper_idx] = profile_y
                finals[object_idx * num_gripper + gripper_idx] = final
                save_gripper_dirs[object_idx * num_gripper + gripper_idx] = save_gripper_dir
        except Exception as e:
            print(e)
            continue
    ray.shutdown()
    gripper_imgs = list(map(lambda x: x[1], sorted(gripper_imgs.items(), key=lambda x: x[0])))
    metrics = list(map(lambda x: x[1], sorted(metrics.items(), key=lambda x: x[0])))
    profiles = list(map(lambda x: x[1], sorted(profiles.items(), key=lambda x: x[0])))
    profiles_x = list(map(lambda x: x[1], sorted(profiles_x.items(), key=lambda x: x[0])))
    profiles_y = list(map(lambda x: x[1], sorted(profiles_y.items(), key=lambda x: x[0])))
    finals = list(map(lambda x: x[1], sorted(finals.items(), key=lambda x: x[0])))
    save_gripper_dirs = list(map(lambda x: x[1], sorted(save_gripper_dirs.items(), key=lambda x: x[0])))
    if render or render_last:
        videos = list(map(lambda x: x[1], sorted(videos.items(), key=lambda x: x[0])))
        return gripper_imgs, metrics, profiles, profiles_x, profiles_y, finals, videos, save_gripper_dirs
    else:
        return gripper_imgs, metrics, profiles, profiles_x, profiles_y, finals, [], save_gripper_dirs