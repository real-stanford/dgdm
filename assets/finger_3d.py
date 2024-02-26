import os
import os
import numpy as np 
from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
import trimesh
import time
import xml.etree.ElementTree as ET

TMP_DIR = './tmp'

def generate_3d_finger_shape(control_points, degree_u=3, degree_v=2, sample_size=100):
    """Generate 3D finger shape from 3D control points.

    Args:
        control_points (np.array): Control points of the finger shape.
        degree_u (int, optional): Degree of the Bezier surface in u-direction. Defaults to 3.
        degree_v (int, optional): Degree of the Bezier surface in v-direction. Defaults to 2.
        sample_size (int, optional): Number of samples. Defaults to 100.
    """
    surf = BSpline.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    surf.set_ctrlpts(control_points, 7, 3)
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)
    surf.sample_size = sample_size
    surf.evaluate()
    os.makedirs(TMP_DIR, exist_ok=True)
    tmp_file = os.path.join(TMP_DIR, '%f.obj' % time.time())
    exchange.export_obj(surf, tmp_file)
    mesh = trimesh.load(tmp_file)
    vertices = mesh.vertices
    faces = mesh.faces
    return vertices, faces

def generate_3d_finger_mesh(control_points, degree_u=3, degree_v=2, sample_size=25, width=0.12):
    surf_vertices, surf_faces = generate_3d_finger_shape(control_points, degree_u, degree_v, sample_size)
    num_surf_vertices = surf_vertices.shape[0]
    all_vertices = np.concatenate([
        surf_vertices,
        surf_vertices + [0, width, 0]
    ])
    surf_contour_indices = np.concatenate([np.arange(sample_size-1), np.arange(sample_size-1, sample_size**2-sample_size, sample_size), np.arange(sample_size**2-1, sample_size**2-sample_size, -1), np.arange(sample_size**2-sample_size, 0, -sample_size)])
    side_faces_upper = np.stack([surf_contour_indices, np.roll(surf_contour_indices, -1), np.roll(surf_contour_indices, -1)+num_surf_vertices], axis=-1)
    side_faces_lower = np.stack([surf_contour_indices, np.roll(surf_contour_indices, -1)+num_surf_vertices, surf_contour_indices+num_surf_vertices], axis=-1)
    bottom_faces = surf_faces + num_surf_vertices
    bottom_faces[:, [1, 2]] = bottom_faces[:, [2, 1]]
    all_faces = np.concatenate([
        surf_faces,
        bottom_faces,
        side_faces_upper,
        side_faces_lower,
    ])
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
    return mesh, surf_vertices

def generate_3d_finger_vertices(control_points, degree_u=3, degree_v=2, sample_size=25):
    surf = BSpline.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    surf.set_ctrlpts(control_points, 7, 3)
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)
    surf.sample_size = sample_size
    return np.array(surf.evalpts).reshape(-1, 3)

def save_3d_gripper(yl, yr, width=0.12, sample_size=25, save_gripper_dir=''):
    x = np.linspace(-0.12, 0.12, 7)
    z = np.linspace(0, 0.12, 3)
    x_n, z_n = np.meshgrid(x, z)
    ctrlpts_l = np.stack([x_n.T.reshape(-1), yl, z_n.T.reshape(-1)], axis=-1)
    ctrlpts_r = np.stack([x_n.T.reshape(-1), yr, z_n.T.reshape(-1)], axis=-1)
    mesh_l, vertices_l = generate_3d_finger_mesh(ctrlpts_l.tolist(), width=width, sample_size=sample_size)
    mesh_r, vertices_r = generate_3d_finger_mesh(ctrlpts_r.tolist(), width=width, sample_size=sample_size)
    os.makedirs(save_gripper_dir, exist_ok=True)
    mesh_l.export(os.path.join(save_gripper_dir, 'fingerl.obj'))
    mesh_r.export(os.path.join(save_gripper_dir, 'fingerr.obj'))
    return np.concatenate((ctrlpts_l, ctrlpts_r), axis=0), np.concatenate((vertices_l, vertices_r), axis=0)

def generate_3d_ctrlpts(yl, yr):
    x = np.linspace(-0.12, 0.12, 7)
    z = np.linspace(0, 0.12, 3)
    x_n, z_n = np.meshgrid(x, z)
    ctrlpts_l = np.stack([x_n.T.reshape(-1), yl, z_n.T.reshape(-1)], axis=-1)
    ctrlpts_r = np.stack([x_n.T.reshape(-1), yr, z_n.T.reshape(-1)], axis=-1)
    return np.concatenate((ctrlpts_l, ctrlpts_r), axis=0)

def generate_3d_gripper(yl, yr, sample_size=25):
    x = np.linspace(-0.12, 0.12, 7)
    z = np.linspace(0, 0.12, 3)
    x_n, z_n = np.meshgrid(x, z)
    ctrlpts_l = np.stack([x_n.T.reshape(-1), yl, z_n.T.reshape(-1)], axis=-1)
    ctrlpts_r = np.stack([x_n.T.reshape(-1), yr, z_n.T.reshape(-1)], axis=-1)
    vertices_l, _ = generate_3d_finger_shape(ctrlpts_l.tolist(), sample_size=sample_size)
    vertices_r, _ = generate_3d_finger_shape(ctrlpts_r.tolist(), sample_size=sample_size)
    return np.concatenate((ctrlpts_l, ctrlpts_r), axis=0), np.concatenate((vertices_l, vertices_r), axis=0)

def create_mesh_elements(num_meshes, mesh_prefix, gripper_idx):
   """ Create mesh elements for a given prefix and number of meshes. """
   return [ET.Element("mesh", name=f"{mesh_prefix}{i:03d}", file=f"grippers/{gripper_idx}/{mesh_prefix}{i:03d}.obj") 
         for i in range(num_meshes)]

def create_geom_elements(num_meshes, mesh_prefix):
   """ Create geom elements for a given prefix and number of meshes. """
   return [ET.Element("geom", mesh=f"{mesh_prefix}{i:03d}", type="mesh", attrib={"class": "collision"})
         for i in range(num_meshes)]

def generate_gripper_3d_xml(left_num_collision_meshes, right_num_collision_meshes, gripper_idx, save_path):
   root = ET.Element("mujoco", model="gripper_3d")
   asset = ET.SubElement(root, "asset")
   # Creating mesh elements for left and right
   left_meshes = create_mesh_elements(left_num_collision_meshes, "fingerl", gripper_idx)
   right_meshes = create_mesh_elements(right_num_collision_meshes, "fingerr", gripper_idx)
   asset.extend([ET.Element("mesh", name="fingerl", file=f"grippers/{gripper_idx}/fingerl.obj"), 
               ET.Element("mesh", name="fingerr", file=f"grippers/{gripper_idx}/fingerr.obj")] + left_meshes + right_meshes)

   default = ET.SubElement(root, "default")

   ET.SubElement(default, "joint", type="slide", axis="0 1 0", damping="1")

   worldbody = ET.SubElement(root, "worldbody")
   fingers = ET.SubElement(worldbody, "body", name="fingers", pos="0 0 0")

   left_jaw = ET.SubElement(fingers, "body", name="left_jaw", pos="0 -0.23 0")
   ET.SubElement(left_jaw, "joint", name="left_grip")
   fingerl = ET.SubElement(left_jaw, "geom", mesh="fingerl", type="mesh", rgba="0.9333 0.7804 0.3490 1")
   fingerl.set("class", "visual")

   for i in range(left_num_collision_meshes):
      fingerl_c = ET.SubElement(left_jaw, "geom", mesh=f"fingerl{i:03d}", type="mesh")
      fingerl_c.set("class", "collision")

   right_jaw = ET.SubElement(fingers, "body", name="right_jaw", pos="0 0.23 0")
   ET.SubElement(right_jaw, "joint", name="right_grip")
   fingerr = ET.SubElement(right_jaw, "geom", mesh="fingerr", type="mesh", rgba="0.6941 0.7647 0.5059 1")
   fingerr.set("class", "visual")
   for i in range(right_num_collision_meshes):
      fingerr_c = ET.SubElement(right_jaw, "geom", mesh=f"fingerr{i:03d}", type="mesh")
      fingerr_c.set("class", "collision")

   actuator = ET.SubElement(root, "actuator")
   left_act = ET.SubElement(actuator, "position", name="left", joint="left_grip")
   left_act.set("ctrlrange", "0 0.1")
   left_act.set("kp", "10")
   right_act = ET.SubElement(actuator, "position", name="right", joint="right_grip")
   right_act.set("ctrlrange", "-0.1 0")
   right_act.set("kp", "10")

   tree = ET.ElementTree(root)
   tree.write(save_path)

def generate_scene_3d_xml(object_idx, gripper_idx, save_path):
    root = ET.Element("mujoco", model="scene")

    defaults = ET.SubElement(root, "default")

    # Add collision default
    collision_default = ET.SubElement(defaults, "default", {"class": "collision"})
    ET.SubElement(collision_default, "geom", group="3", condim="4", friction="1.0 0.005 0.0001")

    # Add visual default
    visual_default = ET.SubElement(defaults, "default", {"class": "visual"})
    ET.SubElement(visual_default, "geom", group="2", contype="0", conaffinity="0")

    # Include external XML files
    ET.SubElement(root, "include", file="object_%d.xml" % object_idx)
    ET.SubElement(root, "include", file="gripper_%d.xml" % gripper_idx)

    # Create worldbody and its child elements
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", name="plane", pos="0 0 -0.01")
    ET.SubElement(body, "geom", type="plane", size="1 1 0.1", rgba="1.0 1.0 1.0 1")

    tree = ET.ElementTree(root)
    tree.write(save_path)