import os
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

def continuous_signed_delta(theta1, theta2):
    delta = theta2 - theta1
    if delta > np.pi:
        delta = delta - 2*np.pi
    elif delta < -np.pi:
        delta = delta + 2*np.pi
    return delta

def sample_pts_from_mesh(mesh_file, num_points=1024):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    pts = np.asarray(pcd.points).reshape((-1, 3))
    return pts

def visualize_finals(finals, save_path):
    plt.clf()
    f = plt.figure(figsize=(10, 6))
    ax = f.add_subplot(111)
    ax.set(ylim=(0, 2*np.pi))
    ax.scatter(np.arange(len(finals)), finals, s=2)
    plt.savefig(save_path)
    plt.close()

def visualize_profile(profile, save_path, ori_range=[-1.0, 1.0]):
    plt.clf()
    signs = np.sign(profile)

    radii = np.array([1])
    thetas = np.linspace(ori_range[0] * np.pi + np.pi, ori_range[1] * np.pi + np.pi, len(profile))
    theta, r = np.meshgrid(thetas, radii)
    u = - 2 * np.pi / len(profile) * np.sin(theta) * signs
    v = 2 * np.pi / len(profile) * np.cos(theta) * signs

    f = plt.figure(figsize=(40, 40))
    ax = f.add_subplot(polar=True)
    ax.quiver(theta, r, u, v, profile, scale=1, width=0.005, headwidth=4, headlength=2, headaxislength=2, cmap='bwr')

    plt.savefig(save_path)
    plt.close()

def visualize_profile_xy_theta(input_ori, input_pos, profile_ori, profile_x, profile_y, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.clf()
    f = plt.figure(figsize=(60, 20))
    ax = f.add_subplot(131, projection='3d')
    ax.set(xlim=(-3, 3), ylim=(-3, 3), zlim=(-1, 1))
    color = np.asarray(['r' if ori == 1 else 'b' if ori == -1 else 'g' for ori in profile_ori])
    x = (input_pos[:, 0]+2.0) * np.cos(input_ori)
    y = (input_pos[:, 0]+2.0) * np.sin(input_ori)
    z = input_pos[:, 1]
    ax.scatter(x, y, z, c=color, s=1)

    ax = f.add_subplot(132, projection='3d')
    ax.set(xlim=(-3, 3), ylim=(-3, 3), zlim=(-1, 1))
    color = np.asarray(['r' if x == 1 else 'b' if x == -1 else 'g' for x in profile_x])
    ax.scatter(x, y, z, c=color, s=1)

    ax = f.add_subplot(133, projection='3d')
    ax.set(xlim=(-3, 3), ylim=(-3, 3), zlim=(-1, 1))
    color = np.asarray(['r' if y == 1 else 'b' if y == -1 else 'g' for y in profile_y])
    ax.scatter(x, y, z, c=color, s=1)
    plt.savefig(os.path.join(save_dir, 'profile.png'))
    plt.close()

def visualize_ctrlpts(ctrlpts, save_path):
    num_pt = ctrlpts.shape[0] // 2
    plt.clf()
    f = plt.figure()
    ax = f.add_subplot(211)
    ax.set(xlim=(-0.12, 0.12), ylim=(-0.045, 0.015))
    ax.scatter(ctrlpts[:num_pt, 0], ctrlpts[:num_pt, 1])
    ax = f.add_subplot(212)
    ax.set(xlim=(-0.12, 0.12), ylim=(-0.045, 0.015))
    ax.scatter(ctrlpts[num_pt:, 0], ctrlpts[num_pt:, 1])
    plt.savefig(save_path)