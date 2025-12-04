import os
from ASGS_help import initialcircle, initialgscpr, generate_blades, parse_images_txt
import numpy as np
import shutil
import re
import matplotlib.pyplot as plt
from PIL import Image
import pyvista as pv
import trimesh
import cv2

def ASGS(scene_name):
    #0.数据准备
    # 0.1背景数据
    # ./data_uav/{scene_name}/{scene_name}_background/images   场景图像:视频抽帧获得
    # ./data_uav/{scene_name}/{scene_name}_background/masks    天空掩码:SAM2算法获得
    # ./data_uav/{scene_name}/{scene_name}_background/sparse   相机-地面位姿:Colmap算法获得(小孔成像模型+序列匹配+txt输出)
    # 0.2无人机数据
    # ./data_uav/{scene_name}/{scene_name}_uav_0/images        无人机图像:场景图像×无人机掩码(执行maskimages(scene_name))
    # ./data_uav/{scene_name}/{scene_name}_uav_0/masks         无人机掩码:SAM2算法获得
    # ./data_uav/{scene_name}/{scene_name}_uav_0/sparse        相机-无人机位姿:Colmap算法获得(小孔成像模型+序列匹配+txt输出):只能匹配部分帧
    # 0.3无人机位姿伪真值
    # ./data_uav/{scene_name}/{scene_name}_pose_gt             相机-无人机位姿:图像2倍抽帧+Colmap算法获得(小孔成像模型+序列匹配+txt输出):部分帧估计结果错误
    # Colmap对无人机位姿估计不准确，需要保证起始帧位姿连续
    shutil.copytree(f'./data_uav/{scene_name}', f'./data/{scene_name}')

    #1.参数确定
    interval = 5            #图像名称间隔
    iteration = 10          #迭代优化重建次数
    match_number = 800      #MASt3R匹配数量阈值
    scene_final = f'{scene_name}_uav_{iteration}'
    number_all = len(os.listdir(f'./data/{scene_name}/{scene_name}_uav_0/images'))
    with open(f'./data/{scene_name}/{scene_name}_uav_0/sparse/0/images.txt', 'r') as file:
        lines = file.readlines()
        number_of_lines = len(lines)
    number_Colmap = int(((number_of_lines - 4)/2 - 1)/interval + 1)
    #2.背景重建
    cmd = f'python ./train.py -s ./data/{scene_name}/{scene_name}_background --out_name ./{scene_name}/{scene_name}_background --mask_type sky'
    os.system(cmd)
    cmd = f'python ./render.py -s ./data/{scene_name}/{scene_name}_background -m ./output/{scene_name}/{scene_name}_background --depth_trunc 100'
    os.system(cmd)

    #3.无人机位姿估计重建
    #3.1初始化用于高斯渲染的文件
    scene_circle = f'{scene_name}_uav_circle'

    #3.2迭代优化重建
    for rec in range(0, iteration):
        scene = f'{scene_name}_uav_{rec}'
        scene_next = f'{scene_name}_uav_{rec + 1}'

        #3.2.1使用现有位姿估计结果进行粗糙重建
        cmd = f'python ./train.py -s ./data/{scene_name}/{scene} --out_name ./{scene_name}/{scene} --mask_type uav'
        os.system(cmd)

        initialcircle(scene_name, scene_circle)
        #3.2.2循环位姿估计
        initialgscpr(scene_name, scene, number_all, number_Colmap, interval)

        for idx in range(number_Colmap * interval + 1, number_all * interval + 1, interval):
            # 基于初始位姿进行渲染
            cmd = f'python render_single.py -s ./data/{scene_name}/{scene_circle} -m output/{scene_name}/{scene} -r1 --id {idx}'
            os.system(cmd)

            # 将渲染结果移动到gscpr文件夹中
            source_file_jpg = f'./output/{scene_name}/{scene}/test/ours_30000/renders/0001.jpg'
            source_file_npy = f'./output/{scene_name}/{scene}/test/ours_30000/vis/0001.npy'
            os.makedirs(f'./gscpr/datasets/{scene_name}/{scene}/rendered', exist_ok=True)
            os.makedirs(f'./gscpr/datasets/{scene_name}/{scene}/rendered_npy', exist_ok=True)
            shutil.copy2(source_file_jpg, os.path.join(f'./gscpr/datasets/{scene_name}/{scene}/rendered', f'{idx:04d}.jpg'))
            shutil.copy2(source_file_npy, os.path.join(f'./gscpr/datasets/{scene_name}/{scene}/rendered_npy', f'{idx:04d}.npy'))

            # 运行gscpr_single进行位姿估计
            cmd = f'conda activate mast3r & python ./gscpr/gscpr_single.py --scene {scene} --scene_name {scene_name}'
            os.system(cmd)

            # 将本帧的位姿估计结果作为下一帧初始位姿
            source_file_txt = f'./gscpr/outputs/{scene_name}/refine_predictions_{scene}/refinew2c_mast3r_{idx:04d}.txt'
            output_file = f'./gscpr/datasets/{scene_name}/{scene}/coarse_poses'
            shutil.copy2(source_file_txt, os.path.join(output_file, f'{idx+interval:04d}.txt'))
            predict_pose_w2c_path = f'./gscpr/datasets/{scene_name}/{scene}/coarse_poses/'
            predict_w2c_ini = np.loadtxt(predict_pose_w2c_path + f'{idx + interval:04d}.txt')
            file_path = f'./data/{scene_name}/{scene_circle}/sparse/0/images.txt'
            new_line_content = str(predict_w2c_ini).replace('[', '').replace(']', '').replace('\n', '')
            lines = '1' + new_line_content + ' 1 0001.jpg' + '\n'
            with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)

        #3.2.3将超过匹配数量阈值的位姿合并入数据集
        shutil.copytree(f'./data/{scene_name}/{scene_name}_uav_0', f'./data/{scene_name}/{scene_next}', dirs_exist_ok=True)
        with open(f'./gscpr/outputs/{scene_name}/logs_{scene}.log', 'r') as file:
            log_data = file.read()
        numbers = re.findall(r'\((\d+),\)', log_data)
        for idx in range(number_Colmap * interval + 1, number_all * interval + 1, interval):
            if int(numbers[int((idx-1)/interval-number_Colmap)]) > match_number:
                refine_pose = np.loadtxt(f'./gscpr/outputs/{scene_name}/refine_predictions_{scene}/refinew2c_mast3r_{idx:04d}.txt')
                file_path = f'./data/{scene_name}/{scene_next}/sparse/0/images.txt'
                new_line_content = str(refine_pose).replace('[', '').replace(']', '').replace('\n', '')
                new_line = '\n' + f'{idx}' + ' ' + new_line_content + ' 1 ' + f'{idx:04d}' + '.jpg' + '\n'
                with open(file_path, 'a', encoding='utf-8') as f:
                            f.writelines(new_line)

        shutil.rmtree(f'./gscpr/datasets/{scene_name}/{scene}/rendered_npy') #911

    #3.3使用最新的位姿估计结果进行精细重建
    cmd = f'python ./train.py -s ./data/{scene_name}/{scene_final} --out_name ./{scene_name}/{scene_final} --mask_type uav'
    os.system(cmd)

    #4.轨迹合成
    #4.1位姿转换
    cameras_background = parse_images_txt(f'./data/{scene_name}/{scene_name}_background/sparse/0/images.txt')
    cameras_uav = parse_images_txt(f'./data/{scene_name}/{scene_final}/sparse/0/images.txt')
    cameras_uav_0 = parse_images_txt(f'./data/{scene_name}/{scene_name}_uav_0/sparse/0/images.txt')#0903
    t_u0, R_u0, img_u0 = cameras_uav[0]
    t_u1, R_u1, img_u1 = cameras_uav_0[(number_Colmap - 1) * interval]#0903
    t_b0, R_b0, img_b0 = cameras_background[0]
    t_b1, R_b1, img_b1 = cameras_background[number_Colmap-1]#0903
    scale = np.linalg.norm(t_u0 - t_u1) / np.linalg.norm(t_b0 - t_b1)
    R = R_b0.T @ R_u0
    traj_points = []
    for i in range(len(cameras_background)):
        for j in range(len(cameras_uav)):
            if cameras_background[i][2] != cameras_uav[j][2]:
                continue
            t_u, R_u, img_u = cameras_uav[j]
            t_b, R_b, img_b = cameras_background[i]
            t_b = t_b * scale#对准坐标尺度，但不大准
            traj_points.append((t_b - R @ t_u)/scale) #829加了负号

    #4.2生成轨迹几何
    spheres = [pv.Sphere(radius=0.05, center=center) for center in traj_points]
    spheres_combined = spheres[0]
    for s in spheres[1:]:
        spheres_combined = spheres_combined.merge(s)
    polyline = pv.lines_from_points(traj_points)
    tube = polyline.tube(radius=0.02)
    final_model = spheres_combined.merge(tube)
    final_model.save(f'./output/{scene_name}/traj.ply')


    #5.桨叶重建
    #5.1高斯重建数据准备
    shutil.copytree(f'./data/{scene_name}/{scene_name}_uav_0/masks', f'./output/{scene_name}/blade/{scene_name}/uav_masks', dirs_exist_ok=True)
    cmd = f'python ./render_filter.py -s ./data/{scene_name}/{scene_final} -m ./output/{scene_name}/{scene_final} -r1'
    os.system(cmd)
    shutil.copytree(f'./output/{scene_name}/{scene_final}/train/ours_30000/alpha', f'./output/{scene_name}/blade/{scene_name}/alpha_masks', dirs_exist_ok=True)
    os.makedirs(f'./output/{scene_name}/blade/{scene_name}/alpha_name_masks', exist_ok=True)
    os.makedirs(f'./output/{scene_name}/blade/{scene_name}/differ_masks', exist_ok=True)
    file_path = f'./data/{scene_name}/{scene_final}/sparse/0/images.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for idx in range(1, number_all + 1):
        try:
            if idx < number_Colmap + 1:
                name = lines[2*(idx - 1)*interval + 4].strip().split()
            else:
                name = lines[2*(idx-number_Colmap) + 2*(number_Colmap - 1)*interval + 4].strip().split()
            name = name[-1]
            source_file_jpg = f'./output/{scene_name}/blade/{scene_name}/alpha_masks/alpha_{idx:04d}.tiff'
            output_file = f"./output/{scene_name}/blade/{scene_name}/alpha_name_masks/{name.replace('jpg', 'tiff')}"
            shutil.copy2(source_file_jpg, output_file)
            img_uav_mask = plt.imread(f'./output/{scene_name}/blade/{scene_name}/uav_masks/{name}')
            img_alpha_name_mask = np.asarray(Image.open(output_file))*255
            img_alpha_name_mask = np.where(img_alpha_name_mask > 1, 255, 0)
            gray_diff = img_uav_mask - img_alpha_name_mask
            gray_diff = np.clip(gray_diff, 0, 255)

            img_sky_mask = cv2.imread(f'./data/{scene_name}/{scene_name}_background/masks/{name}')
            img_sky_mask = cv2.cvtColor(img_sky_mask, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(img_sky_mask, 127, 255, cv2.THRESH_BINARY)
            M = cv2.moments(thresh)
            icyb = int(M["m01"] / M["m00"])
            img_uav_mask = cv2.imread(f'./output/{scene_name}/blade/{scene_name}/uav_masks/{name}')
            img_uav_mask = cv2.cvtColor(img_uav_mask, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(img_uav_mask, 127, 255, cv2.THRESH_BINARY)
            M = cv2.moments(thresh)
            icyu = int(M["m01"] / M["m00"])
            if icyb*2 > icyu and np.sum(img_sky_mask)/255/3840/2160 > 0.01:
                plt.imsave(f'./output/{scene_name}/blade/{scene_name}/differ_masks/{name}', gray_diff, cmap='gray')
        except:
             pass
    shutil.copytree(f'./data/{scene_name}/{scene_final}/sparse', f'./data/{scene_name}/{scene_name}_blade/sparse', dirs_exist_ok=True)
    shutil.copytree(f'./output/{scene_name}/blade/{scene_name}/differ_masks', f'./data/{scene_name}/{scene_name}_blade/images', dirs_exist_ok=True)
    shutil.copytree(f'./output/{scene_name}/blade/{scene_name}/differ_masks', f'./data/{scene_name}/{scene_name}_blade/masks', dirs_exist_ok=True)

    try:
        #5.2桨叶高斯重建
        cmd = f'python ./train.py -s ./data/{scene_name}/{scene_name}_blade --out_name ./{scene_name}/{scene_name}_blade --mask_type blade --iterations 3000'
        os.system(cmd)
        #5.3圆盘拟合
        generate_blades(f'./output/{scene_name}/{scene_name}_blade/point_cloud/iteration_3000/point_cloud.ply', output_mesh_path=f'./output/{scene_name}/blades.ply')
        #场景几何合成
        mesh_uav = trimesh.load(f'./output/{scene_name}/{scene_final}/train/ours_30000/fuse_post.ply')
        mesh_blades = trimesh.load(f'./output/{scene_name}/blades.ply')
        mesh_scene_uav = trimesh.util.concatenate([mesh_uav, mesh_blades])
    except:
        # 场景几何合成
        mesh_uav = trimesh.load(f'./output/{scene_name}/{scene_final}/train/ours_30000/fuse_post.ply')
        mesh_scene_uav = mesh_uav

    shutil.rmtree(f'./output/{scene_name}/{scene_final}/train/ours_30000/alpha')  # 911
    shutil.rmtree(f'./output/{scene_name}/{scene_final}/train/ours_30000/vis')  # 911
    shutil.rmtree(f'./output/{scene_name}/blade/{scene_name}/alpha_masks')  # 911
    shutil.rmtree(f'./output/{scene_name}/blade/{scene_name}/alpha_name_masks')  # 911

    mesh_background = trimesh.load(f'./output/{scene_name}/{scene_name}_background/train/ours_30000/fuse_post.ply')
    transform = np.eye(4)
    transform[:3, :3] = R
    center = mesh_scene_uav.centroid
    mesh_scene_uav.apply_translation(-center)
    mesh_scene_uav.apply_transform(transform)
    mesh_scene_uav.apply_translation(center)
    target_point = np.array(traj_points[0])
    current_center = mesh_scene_uav.centroid
    translation = target_point - current_center
    mesh_scene_uav.apply_translation(translation)
    # 场景几何合成
    mesh_tray = trimesh.load(f'./output/{scene_name}/traj.ply')
    mesh_scene = trimesh.util.concatenate([mesh_background, mesh_scene_uav, mesh_tray])
    mesh_scene_uav.export(f'./output/{scene_name}/uav.ply')
    mesh_scene.export(f'./output/{scene_name}/scene.ply')

    shutil.rmtree(f'./output/{scene_name}/{scene_name}_background')  # 911