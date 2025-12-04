import os
import cv2
import shutil
from PIL import Image
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor

def maskimages(scene_name):
    images_folder = f'E:/AirTarget/ASGSdata/{scene_name}/uav/images'
    masks_folder = f'E:/AirTarget/ASGSdata/{scene_name}/uav/masks'
    output_folder = f'E:/AirTarget/ASGSdata/{scene_name}/uav/uavimages'
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(masks_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        white_bg = np.ones_like(image, dtype=np.uint8) * 255
        masked_image = image * binary_mask[:, :, np.newaxis] + white_bg * (1 - binary_mask[:, :, np.newaxis])
        cv2.imwrite(output_path, masked_image)

def initialcircle(scene_name, scene_circle):
    os.makedirs(f'./data/{scene_name}/{scene_circle}/images', exist_ok=True)
    os.makedirs(f'./data/{scene_name}/{scene_circle}/masks', exist_ok=True)
    white_img = Image.new('RGB', (3840, 2160), color='white')
    white_img.save(f'./data/{scene_name}/{scene_circle}/images/0001.jpg')
    white_img.save(f'./data/{scene_name}/{scene_circle}/masks/0001.jpg')
    shutil.copytree(f'./data/{scene_name}/{scene_name}_uav_0/sparse/0', f'./data/{scene_name}/{scene_circle}/sparse/0', dirs_exist_ok=True)
    with open(f'./data/{scene_name}/{scene_circle}/sparse/0/images.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    last_line = lines[-2].strip()
    parts = last_line.split()
    parts[0] = '1'
    parts[-1] = '0001.jpg'
    last_line = ' '.join(parts) + '\n'
    with open(f'./data/{scene_name}/{scene_circle}/sparse/0/images.txt', 'w', encoding='utf-8') as f:
        f.write(last_line)

def initialgscpr(scene_name, scene, number_all, number_Colmap, interval):
    os.makedirs(f'./gscpr/datasets/{scene_name}/{scene}/focal_length/', exist_ok=True)
    for i in range(number_Colmap * interval + 1, number_all * interval + 1, interval):
        new_name = f"{i:04d}.txt"
        with open(f'./gscpr/datasets/{scene_name}/{scene}/focal_length/{new_name}', 'w', encoding='utf-8') as f:
            f.write('1800')

    shutil.copytree(f'./data/{scene_name}/{scene}/images', f'./gscpr/datasets/{scene_name}/{scene}/query_images', dirs_exist_ok=True)
    files = sorted([f for f in os.listdir(f'./gscpr/datasets/{scene_name}/{scene}/query_images') if os.path.isfile(os.path.join(f'./gscpr/datasets/{scene_name}/{scene}/query_images', f))])
    for file_name in files[:number_Colmap]:
       file_path = os.path.join(f'./gscpr/datasets/{scene_name}/{scene}/query_images', file_name)
       os.remove(file_path)

    shutil.copytree(f'./data/{scene_name}/{scene}/masks', f'./gscpr/datasets/{scene_name}/{scene}/uavmask', dirs_exist_ok=True)
    files = sorted([f for f in os.listdir(f'./gscpr/datasets/{scene_name}/{scene}/uavmask') if os.path.isfile(os.path.join(f'./gscpr/datasets/{scene_name}/{scene}/uavmask', f))])
    for file_name in files[:number_Colmap]:
       file_path = os.path.join(f'./gscpr/datasets/{scene_name}/{scene}/uavmask', file_name)
       os.remove(file_path)

    os.makedirs(f'./gscpr/datasets/{scene_name}/{scene}/gt_poses/', exist_ok=True)
    for i in range(number_Colmap * interval + 1, number_all * interval + 1, interval):
        with open(f'./data_uav/{scene_name}_pose_gt/images.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        last_line = lines[2 + i * 2].strip()
        parts = last_line.split()
        last_line = ' '.join(parts[1:-2]) + '\n'
        new_name = f"{i:04d}.txt"
        with open(f'./gscpr/datasets/{scene_name}/{scene}/gt_poses/{new_name}', 'w', encoding='utf-8') as f:
            f.write(last_line)

    os.makedirs(f'./gscpr/datasets/{scene_name}/{scene}/coarse_poses/', exist_ok=True)
    i = (number_Colmap-1) * interval + 1
    with open(f'./data_uav/{scene_name}_pose_gt/images.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    last_line = lines[2 + i * 2].strip()
    parts = last_line.split()
    last_line = ' '.join(parts[1:-2]) + '\n'
    new_name = f"{(number_Colmap * interval + 1):04d}.txt"
    with open(f'./gscpr/datasets/{scene_name}/{scene}/coarse_poses/{new_name}', 'w', encoding='utf-8') as f:
        f.write(last_line)

def generate_blades(
    input_ply_path,
    output_mesh_path,
    n_clusters=4,
    disk_resolution=60,
    radius_scale=2
):
    # 1. 读取点云
    def load_point_cloud(file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points), pcd

    # 2. KMeans聚类
    def cluster_points(points, n_clusters=n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
        return kmeans.labels_

    # 3. 计算平面法线和圆半径
    def compute_normal_and_radius(points):
        centroid = points.mean(axis=0)
        uu, dd, vv = np.linalg.svd(points - centroid)
        normal = vv[2]

        # 投影到平面，拟合圆
        projected = (points - centroid) - ((points - centroid) @ normal)[:, None] * normal
        projected_2d = projected @ vv[:2].T

        x, y = projected_2d[:, 0], projected_2d[:, 1]
        A = np.c_[2 * x, 2 * y, np.ones(len(x))]
        b = x ** 2 + y ** 2
        cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
        radius = np.sqrt(c + cx**2 + cy**2)

        return normal, radius

    # 4. 创建圆盘网格
    def create_disk_mesh(center, normal, radius, resolution=disk_resolution):
        normal = normal / np.linalg.norm(normal)
        if np.abs(normal[2]) < 0.99:
            ref = np.array([0, 0, 1])
        else:
            ref = np.array([1, 0, 0])
        x_axis = np.cross(normal, ref)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)

        theta = np.linspace(0, 2 * np.pi, resolution)
        circle_points = [center + radius * (np.cos(t) * x_axis + np.sin(t) * y_axis) for t in theta]

        vertices = [center] + circle_points
        triangles = [[0, i, i + 1] for i in range(1, resolution - 1)]
        triangles.append([0, resolution - 1, 1])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        return mesh

    # 主流程
    points, _ = load_point_cloud(input_ply_path)

    # 全局法向量（来自全部点云）
    global_normal, _ = compute_normal_and_radius(points)

    # 聚类
    labels = cluster_points(points)

    # 聚类质心 + 局部半径
    centroids = []
    radii = []

    for i in range(n_clusters):
        cluster_pts = points[labels == i]
        centroid_i = cluster_pts.mean(axis=0)
        _, radius_i = compute_normal_and_radius(cluster_pts)

        centroids.append(centroid_i)
        radii.append(radius_i)

    # 计算平均半径
    avg_radius = np.mean(radii) * radius_scale

    # 生成圆盘（统一半径，统一法向）
    merged_mesh = o3d.geometry.TriangleMesh()
    total_vertices = 0

    for i in range(n_clusters):
        mesh = create_disk_mesh(
            centroids[i],
            global_normal,
            avg_radius
        )

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles) + total_vertices

        merged_mesh.vertices.extend(o3d.utility.Vector3dVector(vertices))
        merged_mesh.triangles.extend(o3d.utility.Vector3iVector(triangles))

        total_vertices += len(vertices)

    merged_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_mesh_path, merged_mesh)


def qvec2rotmat(qvec):
    """将四元数转为旋转矩阵"""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y**2 - 2 * z**2,     2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,       1 - 2 * x**2 - 2 * z**2,   2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,       2 * y * z + 2 * x * w,     1 - 2 * x**2 - 2 * y**2]
    ])

def parse_images_txt(path):
    cameras = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("#") or line.strip() == "":
            continue
        elems = line.strip().split()
        if len(elems) !=10:
            continue  # 跳过 2D 点的行
        image_id = int(elems[0])
        qw, qx, qy, qz = map(float, elems[1:5])
        tx, ty, tz = map(float, elems[5:8])
        image_name = elems[9]
        qvec = [qw, qx, qy, qz]
        tvec = np.array([tx, ty, tz])
        # 世界坐标中相机中心位置
        R = qvec2rotmat(qvec)
        C = -R.T @ tvec  # 相机中心 = -R.T * t
        cameras.append((C, R, image_name))
    return cameras


# def get_max_plane_normal_from_mesh(mesh, num_points=100000, residual_threshold=0.01):
#     # 从网格均匀采样点
#     points = mesh.sample(num_points)
#
#     # RANSAC 拟合平面：拟合 z = ax + by + c
#     XY = points[:, :2]
#     Z = points[:, 2]
#
#     ransac = RANSACRegressor(residual_threshold=residual_threshold)
#     ransac.fit(XY, Z)
#
#     # 平面系数
#     a, b = ransac.estimator_.coef_
#     c = ransac.estimator_.intercept_
#
#     # 平面法向量： [a, b, -1]
#     normal = np.array([a, b, -1.0])
#     normal /= np.linalg.norm(normal)
#
#     return normal

# def rotation_matrix_from_vectors(a, b):
#     a = a / np.linalg.norm(a)
#     b = b / np.linalg.norm(b)
#     v = np.cross(a, b)
#     c = np.dot(a, b)
#     if c < -0.999999:
#         axis = np.array([1, 0, 0]) if abs(a[0]) < 0.99 else np.array([0, 1, 0])
#         v = np.cross(a, axis)
#         v /= np.linalg.norm(v)
#         H = np.eye(3) - 2 * np.outer(v, v)
#         return H
#     s = np.linalg.norm(v)
#     kmat = np.array([[0, -v[2], v[1]],
#                      [v[2], 0, -v[0]],
#                      [-v[1], v[0], 0]])
#     R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
#     return R