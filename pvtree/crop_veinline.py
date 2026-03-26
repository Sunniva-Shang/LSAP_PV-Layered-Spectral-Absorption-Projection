import cv2
import os
import random

def crop(img_path='', case='', new_path=''):
    # img_path = './results/c2.png'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (542, 616))
    (h, w) = img.shape[:2]
    center = (w//2, h//2)
    #rand_ang = random.randint(-3, 3)
    #rand_x = random.randint(-3, 3)
    #rand_y = random.randint(-3, 3)
    if case == 'c1':
        #(380, 380, 3)
        # angler = -9 + rand_ang
        # start_x, start_y = 90+rand_x, 150+rand_y
        # end_x, end_y = 470+rand_x, 530+rand_y
        angler = -26
        start_x, start_y = 80, 100
        end_x, end_y = 450, 480

    elif case == 'c2':
        #(375, 375, 3)
        # angler = -8 + + rand_ang
        # start_x, start_y = 100+rand_x, 145+rand_y
        # end_x, end_y = 475+rand_x, 520+rand_y
        angler = -26
        start_x, start_y = 70, 145
        end_x, end_y = 445, 520
    elif case == 'c3':
        #(345, 345, 3)
        #angler = -16 + rand_ang
        # start_x, start_y = 140+rand_x, 115+rand_y
        # end_x, end_y = 485+rand_x, 460+rand_y

        angler = -14
        start_x, start_y = 100, 115
        end_x, end_y = 445, 460

    elif case == 'c4':
        #(385, 385, 3)
        # angler = -4 + + rand_ang
        # start_x, start_y = 95+rand_x, 135+rand_y
        # end_x, end_y = 480+rand_x, 520+rand_y
        angler = -20
        start_x, start_y = 55, 135
        end_x, end_y = 440, 520

    M = cv2.getRotationMatrix2D(center, angler, 1)
    rotated_image = cv2.warpAffine(img, M, (w, h))

    # start_x, start_y = 90,150
    # end_x, end_y = 470,530
    crop_image = rotated_image[start_y:end_y, start_x:end_x]
    # print(crop_image.shape)
    resized_image = cv2.resize(crop_image, (400,400), interpolation=cv2.INTER_AREA)
    cv2.imwrite(new_path, resized_image)
    
    # cv2.imshow('original img', img)
    # cv2.imshow('rotated img', rotated_image)
    # cv2.imshow('cropped img', crop_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def crop_img(path, newpath):
    # os.makedirs(newpath)
    for di in os.listdir(path):
        imgpath = os.path.join(path, di)
        c = di.split('_')[0]
        dpath = os.path.join(newpath, di)
        crop(imgpath, c, dpath)





if __name__ == '__main__':
    
    #path = './line14/num'
    #newpath = './line14/cropnum'
    # path = './results/deep_results_num70_4000'
    # newpath = './results/crop_deep_results_num70_4000'
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)
    # crop_img(path, newpath)
    crop('./results/newdeep_num70_4000/c4_102_sample0.png', 'c4', './results/test/crop_c4_102_sam0.png')
