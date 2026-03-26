import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from args import parse_args
import random
from mpl_toolkits.mplot3d import Axes3D
from tools import norm, min_distance_point_to_line, show_img_2d, draw_segment2d_rotate
from class_v_s import TPoint, Segment
from pvtree.kamiya import get_random_position_c1, kamiya_optimal
import os, shutil
from crop_veinline import crop
import time
import argparse


def create_finger_position(case):
    if case == 1:
        finger_x = [0, 3, 7, 20, 25, 37, 41, 58, 65, 70]
        finger_z = [52, 68, 71, 76, 80, 80, 80, 76, 71, 50]
    elif case == 2:
        finger_x = [0, 8, 13, 23, 30, 40, 47, 60, 65, 70]
        finger_z = [58, 70, 72, 76, 80, 80, 80, 76, 72, 48]
    elif case == 3:
        finger_x = [0, 4, 9, 18, 26, 36, 43, 58, 65, 70, 53, 65]
        finger_z = [56, 64, 66, 75, 77, 80, 80, 80, 72, 48, 42, 34]
    elif case == 4:
        finger_x = [0, 6, 10, 25, 30, 40, 48, 60, 66, 70]
        finger_z = [56, 68, 72, 75, 80, 80, 80, 75, 70, 50]


    finger_position_list = []
    for i in range(len(finger_x)): 
        bias_x = random.uniform(-1.5, 1.5)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-2, 2)
        finger_position = np.array([finger_x[i] + bias_x, 40 + bias_y, finger_z[i] + bias_z])      
        finger_position_list.append(finger_position)

    return finger_position_list

def create_main_tree_c1(args, segment):
# case1
    list_root_point = []
    
    position_r_x = np.array([33.0, 22.0, 18.0, 25.0, 30.0, 38.0, 46.0, 50.0, 45.0, 38.0])
    position_r_z = np.array([-3.0, 30.0, 42.0, 50.0, 55.0, 58.0, -3.0, 31.0, 45.0, 58.0])
    for i in range(len(position_r_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_r = np.array([position_r_x[i] + bias_x, 40 + bias_y, position_r_z[i] + bias_z])
        if i == 0 or i == 1:
            point_r = TPoint(args, position_r, num_end=9)
        elif i == 2:
            point_r = TPoint(args, position_r, num_end=8)
        elif i == 3:
            point_r = TPoint(args, position_r, num_end=6)
        elif i == 4 or i==8:
            point_r = TPoint(args, position_r, num_end=4)
        elif i == 5 or i==9:
            point_r = TPoint(args, position_r, num_end=2)
        elif i == 6 or i == 7:
            point_r = TPoint(args, position_r, num_end=5)
        
        list_root_point.append(point_r)

    for i in range(1, 6):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    for i in range(7, len(list_root_point)):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    s_finger_point = list_root_point[1: 2] + list_root_point[2: 6] + list_root_point[8:9] + list_root_point[7:8]

    list_branch_finger_point = []

    position_bf_x = np.array([5, 6, 15, 28, 47, 58, 70])
    position_bf_z = np.array([23, 55, 66, 73, 72, 53, 30])
    for i in range(len(position_bf_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_bf = np.array([position_bf_x[i] + bias_x, 40 + bias_y, position_bf_z[i] + bias_z])
        if i == 0 or i == 6:
            point_bf = TPoint(args, position_bf, num_end=1)
        else:
            point_bf = TPoint(args, position_bf, num_end=2)
        list_branch_finger_point.append(point_bf)

    finger_position = create_finger_position(case=1)

    list_move = []
    list_finger_point = []
    for i in range(len(finger_position)):
        finger_point = TPoint(args, finger_position[i])
        if i == 0 or i == 1:
            finger_point.parent = list_branch_finger_point[1]
            list_finger_point.append(finger_point)
            seg1 = Segment(list_branch_finger_point[1], finger_point)
            seg1.index = len(segment)
            segment.append(seg1)
        elif i == 2 or i == 3:
            finger_point.parent = list_branch_finger_point[2]
            list_finger_point.append(finger_point)
            seg2 = Segment(list_branch_finger_point[2], finger_point)
            seg2.index = len(segment)
            segment.append(seg2)

        elif i == 4 or i == 5:
            finger_point.parent = list_branch_finger_point[3]
            list_finger_point.append(finger_point)
            seg3 = Segment(list_branch_finger_point[3], finger_point)
            seg3.index = len(segment)
            segment.append(seg3)

        elif i == 6 or i == 7:
            finger_point.parent = list_branch_finger_point[4]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[4], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 8 or i == 9:
            finger_point.parent = list_branch_finger_point[5]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[5], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        

    list_move = list_branch_finger_point

    for i in range(len(list_move)):
        list_move[i].parent = s_finger_point[i]
        seg = Segment(s_finger_point[i], list_move[i])
        segment.append(seg)

    for i in range(len(segment)):
        segment[i].point_out.radius = args.r_ori + segment[i].point_out.num_end * args.ratioE

def create_main_tree_c2(args, segment):
# case2
    list_root_point = []
    
    position_r_x = np.array([30.0, 25.0, 20.0, 22.0, 33.0, 41.0, 51.0, 50.0, 47.0])
    position_r_z = np.array([-5.0, 24.0, 35.0, 52.0, 58.0, -3.0, 27.0, 39.0, 58.0])
    for i in range(len(position_r_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_r = np.array([position_r_x[i] + bias_x, 40 + bias_y, position_r_z[i] + bias_z])
        if i == 0 or i == 1:
            point_r = TPoint(args, position_r, num_end=7)
        elif i == 2:
            point_r = TPoint(args, position_r, num_end=6)
        elif i == 5 or i == 6:
            point_r = TPoint(args, position_r, num_end=5)    
        elif i == 3 or i == 7:
            point_r = TPoint(args, position_r, num_end=4)
        else:
            point_r = TPoint(args, position_r, num_end=2)
        
        list_root_point.append(point_r)

    for i in range(1, 5):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    for i in range(6, len(list_root_point)):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    s_finger_point = list_root_point[1: 5] + list_root_point[8:9] + list_root_point[7:8] + list_root_point[6:7]

    list_branch_finger_point = []

    position_bf_x = np.array([5, 6, 18, 34, 48, 60, 70])
    position_bf_z = np.array([13, 56, 68, 70, 70, 45, 30])
    for i in range(len(position_bf_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_bf = np.array([position_bf_x[i] + bias_x, 40 + bias_y, position_bf_z[i] + bias_z])
        if i == 0 or i == 6:
            point_bf = TPoint(args, position_bf, num_end=1)
        else:
            point_bf = TPoint(args, position_bf, num_end=2)
        list_branch_finger_point.append(point_bf)

    finger_position = create_finger_position(case=2)

    list_move = []
    list_finger_point = []
    for i in range(len(finger_position)):
        finger_point = TPoint(args, finger_position[i])
        if i == 0 or i == 1:
            finger_point.parent = list_branch_finger_point[1]
            list_finger_point.append(finger_point)
            seg1 = Segment(list_branch_finger_point[1], finger_point)
            seg1.index = len(segment)
            segment.append(seg1)
        elif i == 2 or i == 3:
            finger_point.parent = list_branch_finger_point[2]
            list_finger_point.append(finger_point)
            seg2 = Segment(list_branch_finger_point[2], finger_point)
            seg2.index = len(segment)
            segment.append(seg2)

        elif i == 4 or i == 5:
            finger_point.parent = list_branch_finger_point[3]
            list_finger_point.append(finger_point)
            seg3 = Segment(list_branch_finger_point[3], finger_point)
            seg3.index = len(segment)
            segment.append(seg3)

        elif i == 6 or i == 7:
            finger_point.parent = list_branch_finger_point[4]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[4], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 8 or i == 9:
            finger_point.parent = list_branch_finger_point[5]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[5], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        

    list_move = list_branch_finger_point

    for i in range(len(list_move)):
        list_move[i].parent = s_finger_point[i]
        seg = Segment(s_finger_point[i], list_move[i])
        segment.append(seg)

    for i in range(len(segment)):
        segment[i].point_out.radius = args.r_ori + segment[i].point_out.num_end * args.ratioE

def create_main_tree_c3(args, segment):
# case3
    list_root_point = []
    position_r_x = np.array([32.0, 20.0, 28.0, 36.0, 46.0, 60.0, 42.0, 45.0])
    position_r_z = np.array([-3.0, 40.0, 50.0, 57.0, 60.0, 65.0, -3.0, 25.0])
    for i in range(len(position_r_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_r = np.array([position_r_x[i] + bias_x, 40 + bias_y, position_r_z[i] + bias_z])
        if i == 0 or i == 1:
            point_r = TPoint(args, position_r, num_end=10)
        elif i == 2:
            point_r = TPoint(args, position_r, num_end=8)
        elif i == 3:
            point_r = TPoint(args, position_r, num_end=6)    
        elif i == 4:
            point_r = TPoint(args, position_r, num_end=4)
        else:
            point_r = TPoint(args, position_r, num_end=2)
        
        list_root_point.append(point_r)
   
    for i in range(1, 6):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    for i in range(7, len(list_root_point)):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    s_finger_point = list_root_point[1: 6] + list_root_point[7:8]

    list_branch_finger_point = []
    position_bf_x = np.array([5, 16, 33, 50])
    position_bf_z = np.array([57, 64, 72, 72])
    for i in range(len(position_bf_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_bf = np.array([position_bf_x[i] + bias_x, 40 + bias_y, position_bf_z[i] + bias_z])
        point_bf = TPoint(args, position_bf, num_end=2)
        list_branch_finger_point.append(point_bf)
    list_branch_finger_point.append(s_finger_point[-2])
    list_branch_finger_point.append(s_finger_point[-1])

    finger_position = create_finger_position(case=3)

    list_move = []
    list_finger_point = []
    for i in range(len(finger_position)):
        finger_point = TPoint(args, finger_position[i])
        if i == 0 or i == 1:
            finger_point.parent = list_branch_finger_point[0]
            list_finger_point.append(finger_point)
            seg1 = Segment(list_branch_finger_point[0], finger_point)
            seg1.index = len(segment)
            segment.append(seg1)
        elif i == 2 or i == 3:
            finger_point.parent = list_branch_finger_point[1]
            list_finger_point.append(finger_point)
            seg2 = Segment(list_branch_finger_point[1], finger_point)
            seg2.index = len(segment)
            segment.append(seg2)

        elif i == 4 or i == 5:
            finger_point.parent = list_branch_finger_point[2]
            list_finger_point.append(finger_point)
            seg3 = Segment(list_branch_finger_point[2], finger_point)
            seg3.index = len(segment)
            segment.append(seg3)

        elif i == 6 or i == 7:
            finger_point.parent = list_branch_finger_point[3]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[3], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 8 or i == 9:
            finger_point.parent = list_branch_finger_point[4]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[4], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 10 or i == 11:
            finger_point.parent = list_branch_finger_point[5]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[5], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        

    list_move = list_branch_finger_point[0:4]

    for i in range(len(list_move)):
        list_move[i].parent = s_finger_point[i]
        seg = Segment(s_finger_point[i], list_move[i])
        segment.append(seg)

    for i in range(len(segment)):
        segment[i].point_out.radius = args.r_ori + segment[i].point_out.num_end * args.ratioE

def create_main_tree_c4(args, segment):
# case4
    list_root_point = []
    position_r_x = np.array([30.0, 22.0, 42.0, 45.0, 40.0, 30.0, 42.0])
    position_r_z = np.array([-3.0, 30.0, -3.0, 25.0, 45.0, 50.0, 54.0])
    for i in range(len(position_r_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_r = np.array([position_r_x[i] + bias_x, 40 + bias_y, position_r_z[i] + bias_z])
        if i == 0 or i == 1:
            point_r = TPoint(args, position_r, num_end=1)
        elif i == 2 or i ==3:
            point_r = TPoint(args, position_r, num_end=10)
        elif i == 4:
            point_r = TPoint(args, position_r, num_end=8)    
        else:
            point_r = TPoint(args, position_r, num_end=4)
        
        list_root_point.append(point_r)

    for i in range(1, 2):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    for i in range(3, len(list_root_point)-1):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)
    list_root_point[-1].parent = list_root_point[-3]
    segg = Segment(list_root_point[-3], list_root_point[-1])
    segment.append(segg)

    s_finger_point = list_root_point[1: 2] + list_root_point[5:7] + list_root_point[3: 4]

    list_branch_finger_point = []
    position_bf_x = np.array([0, 19, 20, 36, 50, 57])
    position_bf_z = np.array([38, 60, 70, 74, 72, 53])
    for i in range(len(position_bf_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_bf = np.array([position_bf_x[i] + bias_x, 40 + bias_y, position_bf_z[i] + bias_z])
        if i == 0:
            point_bf = TPoint(args, position_bf)
        else:
            point_bf = TPoint(args, position_bf, num_end=2)
        list_branch_finger_point.append(point_bf)
   
    finger_position = create_finger_position(case=4)

    list_move = []
    list_finger_point = []
    for i in range(len(finger_position)):
        finger_point = TPoint(args, finger_position[i])
        if i == 0 or i == 1:
            finger_point.parent = list_branch_finger_point[1]
            list_finger_point.append(finger_point)
            seg1 = Segment(list_branch_finger_point[1], finger_point)
            seg1.index = len(segment)
            segment.append(seg1)
        elif i == 2 or i == 3:
            finger_point.parent = list_branch_finger_point[2]
            list_finger_point.append(finger_point)
            seg2 = Segment(list_branch_finger_point[2], finger_point)
            seg2.index = len(segment)
            segment.append(seg2)

        elif i == 4 or i == 5:
            finger_point.parent = list_branch_finger_point[3]
            list_finger_point.append(finger_point)
            seg3 = Segment(list_branch_finger_point[3], finger_point)
            seg3.index = len(segment)
            segment.append(seg3)

        elif i == 6 or i == 7:
            finger_point.parent = list_branch_finger_point[4]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[4], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 8 or i == 9:
            finger_point.parent = list_branch_finger_point[5]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[5], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        
    list_move = list_branch_finger_point

    for i in range(len(list_move)):
        if i == 0:
            list_move[i].parent = s_finger_point[0]
            seg = Segment(s_finger_point[i], list_move[i])
        elif i == 1 or i == 2:
            list_move[i].parent = s_finger_point[1]
            seg = Segment(s_finger_point[1], list_move[i])
        elif i == 3 or i == 4:
            list_move[i].parent = s_finger_point[2]
            seg = Segment(s_finger_point[2], list_move[i])
        elif i == 5:
            list_move[i].parent = s_finger_point[3]
            seg = Segment(s_finger_point[3], list_move[i])
        
        segment.append(seg)

    for i in range(len(segment)):
        segment[i].point_out.radius = args.r_ori + segment[i].point_out.num_end * args.ratioE


def save_subfig(fig, ax, save_path):
    ax.axis('tight')
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path, bbox_inches=bbox, pad_inches=0)

def get_mu():
    c_me = random.uniform(0.1, 0.43)
    c_b = random.uniform(0.1, 0.02)
    c_co = random.uniform(0.15, 0.3)
    e = random.uniform(0.004, 0.02)  # 0.009
   
    mu_wt = 14 * e
    mu_me = 10 * e
    mu_b = 8 * e
    mu_co = 1 * e

    x1 = random.uniform(0.85, 1.45) 
    x2 = random.uniform(0.13, 0.17)

    mu1 = (1-c_me) * mu_wt
    mu2 = c_me * mu_me
    mu3 = (1-c_b-c_co) * mu_wt + c_b * mu_b + c_co * mu_co

    y12 = mu1 * x1 + mu2 * x2
    
    return y12, mu3
    

def create_3dtree(args, case=1, fullpath='', croppath='', s=1, num_sams=7):
    
    segment = []
    randnum = random.randint(-10, 10)
    num = 80 + randnum
    
    if case == 1:
        create_main_tree_c1(args, segment)
    elif case == 2:
        create_main_tree_c2(args, segment)
    elif case == 3:
        create_main_tree_c3(args, segment)
    elif case == 4:
        create_main_tree_c4(args, segment)

    for i in range(num):
        position = get_random_position_c1(args)
        new_point = TPoint(args, position, args.r_ori)
        min_seg = min_distance_point_to_line(position, segment)
        kamiya_optimal(args, new_point, min_seg, segment)

    y12, mu3 = get_mu()
    
    for sam in range(num_sams):
        fig = plt.figure(figsize=(7, 8))
        ax = fig.add_subplot(111)
        ang = np.radians(random.randint(-2, 2))
        draw_segment2d_rotate(args, ax, segment, angle=ang, s=s, y12=y12, mu3=mu3) 
        show_img_2d(ax)
        fig.savefig(fullpath+'_sample{}.png'.format(sam), bbox_inches='tight', pad_inches=0)
        plt.close()
        crop(fullpath+'_sample{}.png'.format(sam), 'c{}'.format(case), croppath+'_sample{}.png'.format(sam))
    
    time.sleep(1)

def gen_large_veinline(p, p2, num_id):
    num = num_id // 4
    save = p
    save2 = p2
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.exists(save2):
        os.makedirs(save2)
    for i in tqdm(range(0, num)):
        path1 = os.path.join(save, 'c1_' + str(i))
        path2 = os.path.join(save, 'c2_' + str(i))
        path3 = os.path.join(save, 'c3_' + str(i))
        path4 = os.path.join(save, 'c4_' + str(i))
        path5 = os.path.join(save2, 'c1_' + str(i))
        path6 = os.path.join(save2, 'c2_' + str(i))
        path7 = os.path.join(save2, 'c3_' + str(i))
        path8 = os.path.join(save2, 'c4_' + str(i))
        create_3dtree(case=1, savepath=path1, savepath2=path5,is_black=False, s=i)
        create_3dtree(case=2, savepath=path2, savepath2=path6,is_black=False, s=i)
        create_3dtree(case=3, savepath=path3, savepath2=path7,is_black=False, s=i)
        create_3dtree(case=4, savepath=path4, savepath2=path8,is_black=False, s=i)

def gen_large_veinline_case(args, c, fp, cp, num_id, sams):
    num = num_id
    os.makedirs(fp, exist_ok=True)
    os.makedirs(cp, exist_ok=True)
    for i in tqdm(range(num)):
        s = i
        ffp = os.path.join(fp, f'c{c}_' + str(i))    
        ccp = os.path.join(cp, f'c{c}_' + str(i))   
        create_3dtree(args, case=c, fullpath=ffp, croppath=ccp, num_sams=sams, s=s) 
        time.sleep(1)

def cvt(p):
    for img in os.listdir(p):
        idslist = img.split('_')
        ids = idslist[0]+'_'+idslist[1]
        idpath = os.path.join(p, ids)
        if not os.path.exists(idpath):
            os.mkdir(idpath)
        sp = os.path.join(p, img)
        dp = os.path.join(idpath, img)
        shutil.move(sp, dp)
    
if __name__ == '__main__':

    args = parse_args()
    num_percase = args.num_percase
    num_sams = args.num_sams 
    case = args.case

    full_path = f'./pv_pattern_results/full_c{case}'
    crop_path = args.crop_path

    gen_large_veinline_case(args, case, fp=full_path, cp=crop_path, num_id=num_percase, sams=num_sams)

    

    
    
   
