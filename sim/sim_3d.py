import os
import glob
import sys
from typing import Optional
import shutil

import mujoco
from transforms3d import euler, quaternions
import numpy as np
import subprocess
from mujoco import viewer
import ray
import subprocess

from assets.finger_3d import generate_3d_gripper, save_3d_gripper, generate_gripper_3d_xml, generate_scene_3d_xml
from dynamics.utils import continuous_signed_delta
from assets.scan_object_process import read_object_names, generate_object_3d_xml

OBJECT_DIR = '<directory to 3D object model>/mujoco_scanned_objects/models'

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

def prepare_gripper(gripper_idx: int, model_root: str):
    rs = np.random.RandomState(gripper_idx)
    yl = rs.uniform(-0.1, 0, size=(21))
    yr = rs.uniform(-0.1, 0, size=(21))
    save_gripper_dir = os.path.join(model_root, 'grippers', str(gripper_idx))
    if not os.path.exists(save_gripper_dir):
        ctrlpts, allpts = save_3d_gripper(
            yl,
            yr,
            width=0.1,
            sample_size=25,
            save_gripper_dir=save_gripper_dir,
        )
        meshl_path = os.path.join(save_gripper_dir, "fingerl.obj")
        compute_collision(meshl_path)
        meshr_path = os.path.join(save_gripper_dir, "fingerr.obj")
        compute_collision(meshr_path)
        generate_gripper_3d_xml(len(glob.glob(os.path.join(save_gripper_dir, "fingerl0*.obj"))), len(glob.glob(os.path.join(save_gripper_dir, "fingerr0*.obj"))), gripper_idx, os.path.join(model_root, 'gripper_%d.xml' % gripper_idx))

    else:
        ctrlpts, allpts = generate_3d_gripper(
            yl,
            yr,
            sample_size=25,
        )
    return ctrlpts, allpts

def prepare_object(object_name: str, object_idx: int, model_root: str):
    object_model_dir = os.path.join(OBJECT_DIR, object_name)
    object_model_new = os.path.join(model_root, 'object_%d.xml' % object_idx)
    if not os.path.exists(object_model_new):
        shutil.copytree(object_model_dir, os.path.join(model_root, 'objects', str(object_idx)), dirs_exist_ok = True)
        generate_object_3d_xml(len(glob.glob(os.path.join(model_root, 'objects', str(object_idx), "model_collision_*.obj"))), object_idx, object_model_new)
    return os.path.join(model_root, 'objects', str(object_idx))

# @profile
@ray.remote(num_cpus=2)
def main(model_root, gripper_idx: int=0, object_name: str='BUNNY_RACER', object_idx: int=0, save_dir: str="sim", gui: bool = False):
    ctrlpts, allpts = prepare_gripper(gripper_idx, model_root)
    prepare_object(object_name, object_idx, model_root)
    scene_path = os.path.join(model_root, 'scene_%d_%d.xml' % (object_idx, gripper_idx))
    generate_scene_3d_xml(object_idx, gripper_idx, scene_path)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    reset_qpos = data.qpos.copy()
    reset_qvel = data.qvel.copy()
    reset_force = data.qfrc_applied.copy()
    handle = viewer.launch_passive(model, data) if gui else None

    obj_root_idx = [model.joint(jointid).name for jointid in range(model.njnt)].index(
        "object_root"
    )
    obj_jnt = model.joint(obj_root_idx)
    assert obj_jnt.type == 0  # freejoint

    z_rots = np.arange(0.0, 2 * np.pi, 2 * np.pi / 360)
    x_locs = -0.03+0.06*np.arange(5)/4
    y_locs = -0.03+0.06*np.arange(5)/4
    init_poses = np.zeros((len(z_rots), len(x_locs), len(y_locs), 7))
    final_poses = np.zeros((len(z_rots), len(x_locs), len(y_locs), 7))
    for i, x_loc in enumerate(x_locs):
        for j, y_loc in enumerate(y_locs):
            for k, z_rot in enumerate(z_rots):
                data.qpos[:] = reset_qpos[:]
                data.qvel[:] = reset_qvel[:]
                data.qfrc_applied[:] = reset_force
                data.qpos[obj_jnt.qposadr[0] : obj_jnt.qposadr[0] + 3] = [
                    x_loc,
                    y_loc,
                    0,
                ]
                data.qpos[
                    obj_jnt.qposadr[0] + 3 : obj_jnt.qposadr[0] + 7
                ] = euler.euler2quat(0, 0, z_rot)
                init_poses[k, i, j, :] = data.qpos[
                    obj_jnt.qposadr[0] : obj_jnt.qposadr[0] + 7
                ]
                data.ctrl[0] = 0.5
                data.ctrl[1] = -0.5
                for t in range(800):
                    if handle is not None and t % 10 == 0:
                        handle.sync()
                        input(f"Press Enter to continue..., {t}")
                    mujoco.mj_step(model, data)
                final_poses[k, i, j, :] = data.qpos[
                    obj_jnt.qposadr[0] : obj_jnt.qposadr[0] + 7
                ]
                if not np.isclose(data.qpos[obj_jnt.qposadr[0] + 4], 0.0, atol=1e-2) or not np.isclose(data.qpos[obj_jnt.qposadr[0] + 5], 0.0, atol=1e-2):
                    print("give up: object not upright")
                    return
    save_data = {
        "ctrlpts": ctrlpts,
        "allpts": allpts,
        "object_name": object_name,
        "obj_pos": init_poses[..., :3].reshape((-1, 3)),
        "obj_theta": np.asarray([quaternions.quat2axangle(quat)[-1] for quat in init_poses[..., 3:].reshape((-1, 4))], dtype=np.float32),
        "delta_theta": np.asarray([continuous_signed_delta(quaternions.quat2axangle(last_quat)[-1], quaternions.quat2axangle(quat)[-1]) for last_quat, quat in zip(init_poses[..., 3:].reshape((-1, 4)), final_poses[..., 3:].reshape((-1, 4)))], dtype=np.float32),
        "delta_pos": (final_poses[..., :3] - init_poses[..., :3]).reshape((-1, 3)),
    }
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, "%d_%d.npz" % (object_idx, gripper_idx)), save_data)

if __name__ == "__main__":
    model_root = sys.argv[1]
    gripper_idx = int(sys.argv[2])
    object_idx = int(sys.argv[3])
    num_gripper_parallel = int(sys.argv[4])
    num_object_parallel = int(sys.argv[5])
    save_dir = sys.argv[6]
    num_cpus = int(sys.argv[7])
    object_names = read_object_names()

    ray.init(num_cpus=num_cpus, log_to_driver=False)
    ray_tasks = [main.remote(model_root=model_root, gripper_idx=g_idx, object_name=object_names[object_idx], object_idx=o_idx, save_dir=save_dir, gui=False) for g_idx in range(gripper_idx, gripper_idx+num_gripper_parallel) for o_idx in range(object_idx, object_idx+num_object_parallel)]
    while len(ray_tasks) > 0:
        ready, ray_tasks = ray.wait(ray_tasks, num_returns=1)
        try:
            ray.get(ready[0], timeout=1)
        except Exception as e:
            print(e)
            continue
