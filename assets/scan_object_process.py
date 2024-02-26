import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
import xml.etree.ElementTree as ET

def get_bbox(data_dir):
    max = []
    min = []
    for root, dirs, files in os.walk(data_dir):
        for dir in tqdm(dirs):
            mesh_file = os.path.join(root, dir, 'model.obj')
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            bbox = mesh.get_axis_aligned_bounding_box()
            max.append(bbox.get_max_bound().reshape(-1))
            min.append(bbox.get_min_bound().reshape(-1))
    max = np.stack(max, axis=0)
    # plot histogram
    plt.clf()
    plt.hist(max[..., 0], bins=100)
    plt.savefig('max_x.png')
    plt.clf()
    plt.hist(max[..., 1], bins=100)
    plt.savefig('max_y.png')
    plt.clf()
    plt.hist(max[..., 2], bins=100)
    plt.savefig('max_z.png')
    min = np.stack(min, axis=0)
    plt.clf()
    plt.hist(min[..., 0], bins=100)
    plt.savefig('min_x.png')
    plt.clf()
    plt.hist(min[..., 1], bins=100)
    plt.savefig('min_y.png')
    plt.clf()
    plt.hist(min[..., 2], bins=100)
    plt.savefig('min_z.png')
    print('max: ', np.max(max, axis=0))
    print('min: ', np.min(min, axis=0))

def filter_object(data_dir):
    object_names = []
    for root, dirs, files in os.walk(data_dir):
        for dir in tqdm(dirs):
            mesh_file = os.path.join(root, dir, 'model.obj')
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            bbox = mesh.get_axis_aligned_bounding_box()
            max = bbox.get_max_bound().reshape(-1)
            min = bbox.get_min_bound().reshape(-1)
            if max[0] < 0.1 and min[0] > -0.1 and max[1] < 0.1 and min[1] > -0.1 and max[2] < 0.12:
                object_names.append(dir)
    # save object names
    with open('assets/object_names.txt', 'w') as f:
        for name in object_names:
            f.write(name + '\n')

def read_object_names(test=False):
    filename = 'assets/object_names_test.txt' if test else 'assets/object_names.txt'
    object_names = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            object_names.append(line.strip())
    return object_names

def generate_object_3d_xml(num_collision, object_idx, save_path):
    # Create the root element
    root = ET.Element("mujoco", model="object")

    # Create the 'asset' element
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="object", file="objects/%d/model.obj" % object_idx)

    for i in range(num_collision):
        ET.SubElement(asset, "mesh", name=f"object{i:03d}", file=f"objects/{object_idx}/model_collision_{i}.obj")

    # Create the 'worldbody' element
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", name="object")

    # Add 'freejoint' and 'geom' elements to 'body'
    ET.SubElement(body, "freejoint", name="object_root")
    object_v = ET.SubElement(body, "geom", mesh="object", type="mesh")
    object_v.set("class", "visual")

    for i in range(num_collision):
        object_c = ET.SubElement(body, "geom", mesh=f"object{i:03d}", type="mesh")
        object_c.set("class", "collision")

    # Create an ElementTree object and write to file
    tree = ET.ElementTree(root)
    tree.write(save_path)


if __name__ == '__main__':
    # get_bbox('/store/real/xuxm/mujoco_scanned_objects/models')
    filter_object('/store/real/xuxm/mujoco_scanned_objects/models')