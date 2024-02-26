import os
import cv2
import numpy as np
import trimesh
import triangle

def resample_contour(contour, num_points):
    # Flatten the contour array
    contour = contour.reshape(-1, 2)
    
    # Calculate the distance between each pair of points
    distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    distances = np.insert(distances, 0, 0)  # insert 0 at the start
    cumulative_distances = np.cumsum(distances)
    
    # Create an array of evenly spaced distances along the contour
    uniform_distances = np.linspace(0, cumulative_distances[-1], num_points)
    
    # Use linear interpolation to find the x, y coordinates at the uniform distances
    uniform_contour_x = np.interp(uniform_distances, cumulative_distances, contour[:, 0])
    uniform_contour_y = np.interp(uniform_distances, cumulative_distances, contour[:, 1])
    
    # Stack the coordinates together
    uniform_contour = np.vstack((uniform_contour_x, uniform_contour_y)).T
    uniform_contour = uniform_contour.reshape(-1, 1, 2).astype(np.int32)
    
    return uniform_contour

def extract_contours(image, num_points=100, rescale=True):
    image = cv2.resize(image, (128, 128))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Since we want to apply this to the largest contour, let's first identify it
    contour_lengths = [cv2.arcLength(contour, True) for contour in contours]
    max_contour = contours[np.argmax(contour_lengths)]

    # Resample the largest contour to have a fixed number of points
    resampled_contour = resample_contour(max_contour, num_points)
    # reshape from (num_points, 1, 2) to (num_points, 2)
    resampled_contour = resampled_contour.reshape(-1, 2)

    if rescale:
        # rescale the contour to be in [-0.05, 0.05]
        resampled_contour = resampled_contour / 128 * 0.1 - 0.05

    return resampled_contour

def draw_contour(image):
    contour = extract_contours(image, 100, rescale=False)
    image_with_contour = cv2.resize(image, (128, 128))
    cv2.drawContours(image_with_contour, [contour], -1, (0, 255, 0), 1)
    return contour, image_with_contour

def generate_icon_mesh(img, height, num_points=100):
    contour = extract_contours(img, num_points)
    x = contour[..., 0]
    y = contour[..., 1]
    z = np.zeros_like(x)
    vertices_2d = np.stack([x, y, z], axis=-1)
    # Extrude 
    vertices_3d = np.concatenate([
        vertices_2d,
        vertices_2d + [0, 0, height]
    ])

    # Generate indices for side faces
    indices = np.arange(0, num_points)
    side_faces_upper = np.stack([indices, np.roll(indices, -1) + num_points, np.roll(indices, -1)], axis=1)
    side_faces_lower = np.stack([indices, indices + num_points, np.roll(indices, -1) + num_points], axis=1)
    sides = np.concatenate([side_faces_upper, side_faces_lower])

    # Triangulate top and bottom faces
    # keep the boundary-edges of the triangulation
    top_faces = triangle.triangulate({'vertices': contour, 'segments': np.stack([indices, np.roll(indices, -1)], axis=1)}, 'p')['triangles']
    bottom_faces = top_faces + num_points
    top_faces[:, [1, 2]] = top_faces[:, [2, 1]]

    # Combine faces
    faces_3d = np.concatenate([sides, top_faces, bottom_faces])

    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces_3d) 

    return mesh, contour

def save_icon_mesh(img, height, num_points, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    mesh, contour = generate_icon_mesh(img, height, num_points)
    mesh_path = os.path.join(save_dir, 'object.obj')
    mesh.export(mesh_path)
    return contour, mesh_path