import json
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, quaternion_apply



# Load the SMPL-X .npz file
npz_file = "SMPLX_FEMALE.npz"
smplx_data = np.load(npz_file)

# Extract vertex positions and face indices
smplx_vertices = smplx_data["v_template"]  # Shape: (10475, 3)
smplx_faces = smplx_data["f"]              # Shape: (20908, 3)

# Save the data for further processing
np.savetxt("smplx_vertices.txt", smplx_vertices)
np.savetxt("smplx_faces.txt", smplx_faces, fmt="%d")

print("SMPL-X vertices and faces extracted.")





# Load the JSON file
file_path = "state_dict_0813.json"

subject_id = file_path.split('_')[2]
subject_id = subject_id.split('.')[0]
try:
    with open(file_path, 'r') as f:
        test_dict = json.load(f)
    # Display a preview of the data
    test_dict
except FileNotFoundError:
    "The file 'test_dict.json' was not found."
except json.JSONDecodeError as e:
    f"The file could not be decoded: {e}"





faces = np.loadtxt('smplx_faces.txt', dtype=int)  # Load faces (indices of vertices)
# vertices = np.loadtxt('smplx_vertices.txt')  # Load vertices
vertices = np.loadtxt('smplx_vertices.txt')  # Load vertices

gaussian_to_face = test_dict['_gaussian_to_face']

# Convert data to PyTorch tensors
faces_tensor = torch.tensor(faces, dtype=torch.long)
vertices_tensor = torch.tensor(vertices, dtype=torch.float32)

gaussian_to_face_tensor = torch.tensor(gaussian_to_face, dtype=torch.long)





# Function to calculate face transformations
def calc_faces_transform(vertices, faces):
    # print(vertices[faces])
    T = torch.mean(vertices[faces], dim=1)

    sampled = vertices[faces]
    print(sampled)
    vec1 = sampled[:, 2] - sampled[:, 1]

    vec2 = sampled[:, 0] - sampled[:, 1]
    vec3 = sampled[:, 0] - sampled[:, 2]
    cross = torch.cross(vec1, vec2, dim=-1)
    # print(cross)
    norm = torch.nn.functional.normalize(cross, eps=1e-6, dim=-1)
    vec1 = torch.nn.functional.normalize(vec1, eps=1e-6, dim=-1)
    prod = torch.cross(vec1, norm, dim=-1)
    prod = torch.nn.functional.normalize(prod, eps=1e-6, dim=-1)
    rotmat = torch.permute(torch.stack([vec1, norm, prod]), (1, 0, 2))
    rotmat = torch.transpose(rotmat, 1, 2)
    R = matrix_to_quaternion(rotmat)

    MAX_SCALE = 0.05
    area = torch.norm(cross, p=2, dim=-1, keepdim=True)
    vec3_length = torch.norm(vec3, p=2, dim=-1, keepdim=True)
    h = area / vec3_length
    k = torch.mean(torch.stack([h, vec3_length]), dim=0) / MAX_SCALE
    return T, R, k, h
def quaternion_raw_multiply_test(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)
from plyfile import PlyData, PlyElement
import numpy as np
import torch
def save_ply( path,xyz,color,opacity,scaling,rotation):
        # add on code , check shape
        # initalize the sh feature, but not trained, for the sake of saving ply
        N = xyz.shape[0]  # Number of points
        max_sh_degree = 3

        # Initialize self._features_dc with shape (N, 1, 3) filled with zeros
        features_dc = color.unsqueeze(1)
        features_dc = features_dc.clone()
        # Initialize self._features_rest with shape (N, 3, 24) filled with zeros
        num_sh_coeffs = (max_sh_degree + 1) ** 2 - 1  # 15 for max_sh_degree=4
        features_rest = torch.zeros((N, 3, num_sh_coeffs), dtype=torch.float32, device="cuda")
        features_rest = features_rest
        print('features_dc :', features_dc.shape)
        print('features_rest :', features_rest.shape)

        # original code
        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scale = scaling.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(
            features_dc, features_rest, scaling, rotation
        )]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

def construct_list_of_attributes(_features_dc,_features_rest,_scaling,_rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(_features_dc.shape[1]*_features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(_features_rest.shape[1]*_features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(_rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
def s_act(x, min_s_value, max_s_value):
    if isinstance(x, float):
        x = torch.tensor(x).squeeze()
    return min_s_value + torch.sigmoid(x) * (max_s_value - min_s_value)

# Process Gaussian data
batch_size = 1  # Assume a single batch
xyz_list = []
rotation_list = []
scaling_list = []

for frame_id in range(batch_size):
    # Compute transformations
    T, R, k,cross= calc_faces_transform(vertices_tensor, faces_tensor)
    T_ori = T
#     # Map Gaussian indices
    T = T[gaussian_to_face_tensor]
    R = R[gaussian_to_face_tensor]
    k = k[gaussian_to_face_tensor]
    cross=cross[gaussian_to_face_tensor]


scaling = torch.tensor(test_dict['_scaling'])
sig = torch.sigmoid(scaling)
T_np = T.numpy()
R_np = R.numpy()
k_np = k.numpy()
cross_np = cross.numpy()
# Save as separate text files
np.savetxt("T_values.txt", T_np, fmt="%.6f", header="Translations (T)")
np.savetxt("R_values.txt", R_np, fmt="%.6f", header="Rotations (R - quaternions)")
np.savetxt("k_values.txt", k_np, fmt="%.6f", header="Scaling factors (k)")
# Notify about saving


origianl_xyz = torch.tensor(test_dict['_xyz'])

updated_xyz = T + quaternion_apply(R, origianl_xyz) * k
# updated_xyz[:, 0] = -updated_xyz[:, 0]  # Negate x-coordinates

t = s_act(scaling, 0, 0.1)
updated_scale = t * k
def q_normalize(q):
    """
    Normalize the coefficients of a given quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4

    norm = torch.sqrt(torch.sum(torch.square(q), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
    assert not torch.any(
        torch.isclose(norm, torch.zeros_like(norm, device=q.device)))  # check for singularities
    return torch.div(q, norm[:, None])  # q_norm = q / ||q||

rotation = q_normalize(torch.tensor(test_dict['_rotation']))
rigid_rotation = quaternion_multiply(R, rotation)
updated_rotation = rigid_rotation

# mirror
updated_rotation[:,0] *= -1
updated_rotation[:,1] *= -1
updated_xyz[:,0] *= -1


np.savetxt("temp_values.txt", gaussian_to_face_tensor.numpy(), fmt="%.6f", header="temp")


# ******************************************************** Mesh Face Centroid info ****************

import numpy as np
import trimesh  # Install using `pip install trimesh`
from PIL import Image
from scipy.spatial.transform import Rotation as R  # For quaternion calculation

# Load the mesh
mesh = trimesh.load("temp.obj", process=False)

# Load the texture (image)
texture = Image.open("extracted_texture.png")
texture_pixels = np.array(texture)
texture_height, texture_width, _ = texture_pixels.shape

# Initialize lists to store properties for all faces
face_centroids = []
face_normals = []
face_colors = []
face_areas = []
face_quaternions = []
face_scales = []

from scipy.spatial.transform import Rotation as R

for face_idx, face in enumerate(mesh.faces):

    # Get the vertices of the face
    vertices = mesh.vertices[face]
    #print(vertices)
    # Compute the centroid (mean)
    centroid = np.mean(vertices, axis=0)
    face_centroids.append(centroid)
    
    # Compute the normal vector
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    face_normals.append(normal)
    
    v0, v1, v2 = mesh.vertices[face]
        # Compute edge lengths
    edge_lengths = [
        np.linalg.norm(v1 - v0),
        np.linalg.norm(v2 - v1),
        np.linalg.norm(v0 - v2),
    ]
    
    # Find the longest edge
    longest_edge_idx = np.argmax(edge_lengths)

    if longest_edge_idx == 0:
        base_start, base_end, apex = v0, v1, v2
    elif longest_edge_idx == 1:
        base_start, base_end, apex = v1, v2, v0
    else:
        base_start, base_end, apex = v2, v0, v1
        # Compute the longest edge length
    longest_edge = edge_lengths[longest_edge_idx]
    
    # Compute the shortest perpendicular (height)
    height = np.linalg.norm(np.cross(base_end - base_start, apex - base_start)) / longest_edge
    
    # longest_edge_vector = base_end - base_start
    # x_axis = longest_edge_vector / np.linalg.norm(longest_edge_vector)
    
    # # Compute the shortest perpendicular height direction (y-axis)
    # perpendicular_vector = np.cross(longest_edge_vector, apex - base_start)
    # y_axis = perpendicular_vector / np.linalg.norm(perpendicular_vector)
    
    # # Compute the normal to the face (z-axis)
    # normal = np.cross(longest_edge_vector, perpendicular_vector)
    # z_axis = normal / np.linalg.norm(normal)
    
    # # Construct the rotation matrix
    # rotation_matrix = np.stack([x_axis, z_axis,y_axis], axis=1)
    
    # # Convert to quaternion
    # quaternion = R.from_matrix(rotation_matrix).as_quat()
    sampled = mesh.vertices[face]
    vec1 = sampled[:, 2] - sampled[:, 1]
    vec2 = sampled[:, 0] - sampled[:, 1]
    vec3 = sampled[:, 0] - sampled[:, 2]
    
    vec1 = torch.tensor(vec1)
    vec2 = torch.tensor(vec2)
    vec3 = torch.tensor(vec3)
    cross = torch.cross(vec1, vec2, dim=-1)
    norm = torch.nn.functional.normalize(cross, eps=1e-6, dim=-1)
 
    vec1 = torch.nn.functional.normalize(vec1, eps=1e-6, dim=-1)
    prod = torch.cross(vec1, norm, dim=-1)
    prod = torch.nn.functional.normalize(prod, eps=1e-6, dim=-1)
    #print(torch.stack([vec1, norm, prod]).shape)
    rotmat = torch.stack([vec1, norm, prod])  # Shape: (3, 3)
    # rotmat = rotmat.T
    R = matrix_to_quaternion(rotmat)  
    
    # def compute_face_quaternion(normal):
    #     # Compute the quaternion aligning the face normal
    #     default_up = np.array([0, 0, 1])  # Default quad orientation
    #     normal = normal / np.linalg.norm(normal)
    #     axis = np.cross(default_up, normal)
    #     angle = np.arccos(np.dot(default_up, normal))
    #     if np.linalg.norm(axis) < 1e-6:  # Handle collinear vectors
    #         face_quaternion = np.array([1, 0, 0, 0]) if angle < 1e-6 else np.array([0, 1, 0, 0])
    #     else:
    #         axis = axis / np.linalg.norm(axis)
    #         face_quaternion = R.from_rotvec(axis * angle).as_quat()  # [x, y, z, w]

    #     combined_quaternion =  R.from_quat(face_quaternion).as_matrix()

    #     return R.from_matrix(combined_quaternion).as_quat()
    
    # quaternion = compute_face_quaternion(normal)

    face_quaternions.append(np.array(R))
    
    # Get the UV coordinates for the face
    uv_coords = mesh.visual.uv[face]  # UV coordinates for the face
    
    # Sample the texture at each UV coordinate
    face_color = []
    for uv in uv_coords:
        u, v = uv
        pixel_x = int(u * texture_width)
        pixel_y = int((1 - v) * texture_height)  # Flip V-coordinate
        color = texture_pixels[pixel_y, pixel_x, :3]
        face_color.append(color)
    
    # Compute the average color for the face
    average_color = np.mean(face_color, axis=0)
    face_colors.append(average_color)
    
    # # Compute the rotation as a quaternion
    # # Basis vectors: u (v1 normalized), v (v2 orthogonalized), w (normal)
    # u = v1 / np.linalg.norm(v1)
  
    # v = np.cross(normal, u)

    # # Create rotation matrix
    # rotation_matrix = np.stack([u, v, normal], axis=1)
    
    # # Convert rotation matrix to quaternion
    # quaternion = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
    # face_quaternions.append(quaternion)
    

  
    
    vertices = mesh.vertices[face]

    face_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0
    z_scale = np.sqrt(face_area) / 10.0
    face_scales.append([ 0.01,0.01, z_scale])


# Convert lists to arrays for easier manipulation
face_centroids = np.array(face_centroids)
face_normals = np.array(face_normals)
face_colors = np.array(face_colors) / 255  # Normalize colors to [0, 1]
face_areas = np.array(face_areas)
face_quaternions = np.array(face_quaternions)
print(face_quaternions.shape)
face_scales = np.array(face_scales)

face_quaternions[:,0] *= -1
face_quaternions[:,1] *= -1
face_centroids[:,0] *= -1





# Print some summaries
print(f"Processed {len(mesh.faces)} faces.")
print("Centroids:", face_centroids[:5])      # Print first 5 centroids
print("Normals:", face_normals[:5])          # Print first 5 normals
print("Colors:", face_colors[:5])            # Print first 5 colors
print("Areas:", face_areas[:5])              # Print first 5 areas
print("Quaternions:", face_quaternions[:5])  # Print first 5 quaternions
print("Scales:", face_scales[:5])            # Print first 5 scales




print(face_scales[:5])
print(torch.tensor(updated_scale)[:5])




#############################  Combined PLY *************************************************
# Add the priority as the first column
combined_xyz = np.vstack((face_centroids, updated_xyz)) 

combined_rotation = np.vstack((face_quaternions, updated_rotation ))
combined_colors = np.vstack(( face_colors, test_dict['_color']))

combined_opacity =  np.vstack(( np.ones((len(face_normals),1))  , test_dict['_opacity']))
combined_scaling = np.vstack((face_scales, updated_scale  ))

out_ply_name = subject_id +'.ply'
save_ply( out_ply_name,xyz=updated_xyz,
          color = torch.tensor(test_dict['_color']),
          opacity = torch.tensor(test_dict['_opacity']),
          scaling = updated_scale,
          rotation = updated_rotation)
#save_ply( 'test_mesh_only.ply',xyz=torch.tensor(face_centroids),
#          color = torch.tensor(face_colors),
#          opacity = torch.tensor(np.ones((len(face_normals),1))),
#          scaling = torch.tensor(face_scales),
#          rotation =  torch.tensor(face_quaternions)
#          )
# save_ply( 'test1.ply',xyz=torch.tensor(combined_xyz),
#           color = torch.tensor(combined_colors),
#           opacity = torch.tensor(combined_opacity),
#           scaling = torch.tensor(combined_scaling),
#           rotation =  torch.tensor(combined_rotation)
#           )
