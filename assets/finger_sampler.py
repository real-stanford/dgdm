import os
import numpy as np 
from scipy.interpolate import CubicSpline
import trimesh
import xml.etree.ElementTree as ET

def generate_finger_shape(x, y, width, height, num_points=100):
   # Create spline (cubic curve, also degree=3 b-spline)
   cs = CubicSpline(x, y)
   x_new = np.linspace(x.min(), x.max(), num_points)
   y_new = cs(x_new)
   z = np.zeros_like(x_new)
   vertices_2d = np.stack([x_new, y_new, z], axis=-1)

   # Extrude 
   vertices_3d = np.concatenate([
      vertices_2d,
      vertices_2d + [0, width, 0],  
      vertices_2d + [0, width, height],
      vertices_2d + [0, 0, height]
   ])

   # Create faces
   bottom = [[i+num_points, i+num_points+1, i+1, i] for i in range(num_points-1)]
   top = [[i+2*num_points, i+3*num_points, i+3*num_points+1, i+2*num_points+1] for i in range(num_points-1)]
   left = [[i, i+1, i+3*num_points+1, i+3*num_points] for i in range(num_points-1)]
   right = [[i+2*num_points, i+2*num_points+1, i+num_points+1, i+num_points] for i in range(num_points-1)]
   front = [[3*num_points, 2*num_points, num_points, 0]]
   back = [[num_points-1, 2*num_points-1, 3*num_points-1, 4*num_points-1]]

   faces_3d = left + right + front + back + top + bottom

   # Create mesh
   mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces_3d) 

   return mesh, x_new, y_new

def generate_gripper(finger_x, finger_yl, finger_yr, num_points):
   cs_l = CubicSpline(finger_x, finger_yl)
   x_new = np.linspace(finger_x.min(), finger_x.max(), num_points)
   y_new_l = cs_l(x_new)
   cs_r = CubicSpline(finger_x, finger_yr)
   y_new_r = cs_r(x_new)
   ctrlptsl = np.stack([finger_x, finger_yl], axis=-1)
   ctrlptsr = np.stack([finger_x, finger_yr], axis=-1)
   ctrlpts = np.concatenate((ctrlptsl, ctrlptsr), axis=0)
   allptsl = np.stack([x_new, y_new_l], axis=-1)
   allptsr = np.stack([x_new, y_new_r], axis=-1)
   allpts = np.concatenate((allptsl, allptsr), axis=0)
   return ctrlpts, allpts

def save_gripper(finger_x, finger_yl, finger_yr, width, height, num_points, save_gripper_dir):
    os.makedirs(save_gripper_dir, exist_ok=True)
    meshl, x_new_l, y_new_l = generate_finger_shape(finger_x, finger_yl, width, height, num_points)
    meshl.export(os.path.join(save_gripper_dir, 'fingerl.obj'))
    meshr, x_new_r, y_new_r = generate_finger_shape(finger_x, finger_yr, width, height, num_points) 
    meshr.export(os.path.join(save_gripper_dir, 'fingerr.obj'))
    ctrlptsl = np.stack([finger_x, finger_yl], axis=-1)
    ctrlptsr = np.stack([finger_x, finger_yr], axis=-1)
    ctrlpts = np.concatenate((ctrlptsl, ctrlptsr), axis=0)
    allptsl = np.stack([x_new_l, y_new_l], axis=-1)
    allptsr = np.stack([x_new_r, y_new_r], axis=-1)
    allpts = np.concatenate((allptsl, allptsr), axis=0)
    return ctrlpts, allpts

def create_mesh_elements(num_meshes, mesh_prefix, gripper_idx):
   """ Create mesh elements for a given prefix and number of meshes. """
   return [ET.Element("mesh", name=f"{mesh_prefix}{i:03d}", file=f"grippers/{gripper_idx}/{mesh_prefix}{i:03d}.obj") 
         for i in range(num_meshes)]

def create_geom_elements(num_meshes, mesh_prefix):
   """ Create geom elements for a given prefix and number of meshes. """
   return [ET.Element("geom", mesh=f"{mesh_prefix}{i:03d}", type="mesh", attrib={"class": "collision"})
         for i in range(num_meshes)]

def generate_xml_optimized(left_num_collision_meshes, right_num_collision_meshes, gripper_idx, save_path):
    root = ET.Element("mujoco", model="gripper_2d")
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

    # Left jaw and its geometries
    left_jaw = ET.SubElement(fingers, "body", name="left_jaw", pos="0 -0.15 0")
    ET.SubElement(left_jaw, "joint", name="left_grip")
    fingerl = ET.SubElement(left_jaw, "geom", mesh="fingerl", type="mesh", attrib={"class": "visual"})
    left_jaw.extend(create_geom_elements(left_num_collision_meshes, "fingerl"))

    # Right jaw and its geometries
    right_jaw = ET.SubElement(fingers, "body", name="right_jaw", pos="0 0.15 0")
    ET.SubElement(right_jaw, "joint", name="right_grip")
    fingerr = ET.SubElement(right_jaw, "geom", mesh="fingerr", type="mesh", attrib={"class": "visual"})
    right_jaw.extend(create_geom_elements(right_num_collision_meshes, "fingerr"))

    actuator = ET.SubElement(root, "actuator")
    left_act = ET.SubElement(actuator, "position", name="left", joint="left_grip", ctrlrange="0 0.1", kp="10")
    right_act = ET.SubElement(actuator, "position", name="right", joint="right_grip", ctrlrange="-0.1 0", kp="10")

    tree = ET.ElementTree(root)
    tree.write(save_path)
    
def generate_xml(left_num_collision_meshes, right_num_collision_meshes, gripper_idx, save_path):
   root = ET.Element("mujoco", model="gripper_2d")
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

   left_jaw = ET.SubElement(fingers, "body", name="left_jaw", pos="0 -0.15 0")
   ET.SubElement(left_jaw, "joint", name="left_grip")
   fingerl = ET.SubElement(left_jaw, "geom", mesh="fingerl", type="mesh")
   fingerl.set("class", "visual")

   for i in range(left_num_collision_meshes):
      fingerl_c = ET.SubElement(left_jaw, "geom", mesh=f"fingerl{i:03d}", type="mesh")
      fingerl_c.set("class", "collision")

   right_jaw = ET.SubElement(fingers, "body", name="right_jaw", pos="0 0.15 0")
   ET.SubElement(right_jaw, "joint", name="right_grip")
   fingerr = ET.SubElement(right_jaw, "geom", mesh="fingerr", type="mesh")
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

def generate_scene_xml(object_idx, gripper_idx, save_path):
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