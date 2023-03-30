import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from aicsimageio import AICSImage
import open3d as o3d
import alphashape
import pyvista as pv

# Load image
image_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphMap/data/yx1_samples/20230322/RT/1A_LM010_RT_kikume.nd2"
imObject = AICSImage(image_path)

# get resolution
res_raw = imObject.physical_pixel_sizes
res_array = np.asarray(res_raw)
res_array = np.insert(res_array, 0, 1)
pixel_size_z = res_array[1]
pixel_size_x = res_array[2]
pixel_size_y = res_array[3]

# find brightest pixel
imData = np.squeeze(imObject.data)
max_pos_z = np.argmax(imData, axis=0)
max_brightness_z = np.max(imData, axis=0)

# generate x and y axes
xg, yg = np.meshgrid(range(max_pos_z.shape[1]), range(max_pos_z.shape[0]))

im95 = np.percentile(max_brightness_z, 90)
x_plot = xg[np.where(max_brightness_z >= im95)] * pixel_size_x
y_plot = yg[np.where(max_brightness_z >= im95)] * pixel_size_y
z_plot = max_pos_z[np.where(max_brightness_z >= im95)] * pixel_size_z

# downsample
n_samples = 25000
index_vec = range(0, x_plot.size)
mesh_indices = np.random.choice(index_vec, n_samples)

x_plot_sz = x_plot
y_plot_sz = y_plot
z_plot_sz = z_plot

# convert xyz coordinates to a point cloud object
xyz_array = np.concatenate((np.reshape(x_plot_sz[mesh_indices], (n_samples, 1)),
                            np.reshape(y_plot_sz[mesh_indices], (n_samples, 1)),
                            np.reshape(z_plot_sz[mesh_indices], (n_samples, 1))), axis=1)


point_cloud = pv.PolyData(xyz_array)

surf = point_cloud.delaunay_2d()
surf.plot(show_edges=True)

# pcd.points = o3d.utility.Vector3dVector(xyz_array)
#
# # estimate normals
# pcd.estimate_normals()
#
# # use Poisson method to estimate mesh surface
# #
#
# # distances = pcd.compute_nearest_neighbor_distance()
# # avg_dist = np.mean(distances)
#
# # pcd_ds = pcd.voxel_down_sample(voxel_size=25)
#
# # radius = [20]
# # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
# #     pcd, o3d.utility.DoubleVector(radius))
# #
# # poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
# #                                                                 pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
# # bbox = pcd.get_axis_aligned_bounding_box()
# # p_mesh_crop = poisson_mesh.crop(bbox)
# tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
# alpha = 0.5
# # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
# #         pcd, alpha, tetra_mesh, pt_map)
#
# mesh = alphashape.alphashape(xyz_array, alpha)
# # mesh.compute_vertex_normals()
# # alpha_fish.show()
# o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True)
#
# fig = go.Figure()
# fig.add_trace(go.Mesh3d(x=xyz_array[:, 0], y=xyz_array[:, 1], z=xyz_array[:, 2],
#                                     alphahull=9,
#                                     opacity=0.25,
#                                     color='gray'))
#
# fig.show()