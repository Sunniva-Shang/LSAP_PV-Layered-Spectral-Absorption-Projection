
import os, shutil
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from args import parse_args
import random

def blend(vein_path,palm_path, newpath):
    img1 = cv2.imread(vein_path)
    img2 = cv2.imread(palm_path)
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    h, w, c = img1.shape
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img1,img1, mask = mask)
    img2_fg = cv2.bitwise_and(img2,img2, mask = mask_inv)
    dst = cv2.add(img1_bg, img2_fg)
    cv2.imwrite(newpath, dst)

def blend_folder(vein_path, palmprint_path, pv_path):
    '''
        Blend palmprint lines and palm vein patterns.
    '''
    list_v = os.listdir(vein_path)
    list_p = os.listdir(palmprint_path)

    for i in tqdm(range(len(list_v))):
        idsvein = os.path.join(vein_path, list_v[i])
        idspalm = os.path.join(palmprint_path, list_p[i])
        idspv =  os.path.join(pv_path, list_v[i])
        os.makedirs(idspv, exist_ok=True)

        imgveinlist =  os.listdir(idsvein)
        imgpalmlist =  os.listdir(idspalm)

        for j in range(len(imgveinlist)):
            imgvein =  os.path.join(idsvein, imgveinlist[j])
            imgpalm =  os.path.join(idspalm, imgpalmlist[j])
            imgpv =  os.path.join(idspv, imgveinlist[j])
            blend(imgvein, imgpalm, imgpv)

def aug(path, newpath):

    transfor = transforms.Compose([

        transforms.Resize((265,265), interpolation=Image.BILINEAR),
        transforms.RandomPerspective(distortion_scale=0.1, p=1, fill=255),
        transforms.RandomCrop((256,256), padding=0, pad_if_needed=False),
    ])
    
    for ids in tqdm(os.listdir(path)):
        idp = os.path.join(path, ids)
        nidp = os.path.join(newpath, ids)
        os.makedirs(nidp, exist_ok=True)
        for imgs in os.listdir(idp):
            imgp = os.path.join(idp, imgs)
            nimgp = os.path.join(nidp, imgs)
            img = Image.open(imgp)
            sam = transfor(img)
            sam.save(nimgp)

def era_aug(path, newpath, sams):

    for i in tqdm(os.listdir(path)):
        imgpath = os.path.join(path, i)
        era(imgpath, newpath, sams=sams)

def era(imgpath, newpath, sams=20):
    transfor2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.9, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=1),
        transforms.ToPILImage(), 
    ])
    img = Image.open(imgpath)
    ids = imgpath.split('/')[-1].split('.')[0] 
    subpath = os.path.join(newpath, ids)
    os.makedirs(subpath, exist_ok=True)

    for i in range(sams):
        newimgpath = os.path.join(subpath, ids+'_sample'+'{}.png'.format(i))
        sam = transfor2(img)
        sam.save(newimgpath)

def cvt(p):
    for img in os.listdir(p):
        imgpath = os.path.join(p, img)
        if os.path.isdir(imgpath):
            pass
        else:
            case, ids, _ = img.split('_')
            idname = case + '_' + ids
            idpath = os.path.join(p, idname)
            os.makedirs(idpath, exist_ok=True)
            newimgpath = os.path.join(idpath, img)
            shutil.move(imgpath, newimgpath)

def ChangeIDName(p):
    for ids in os.listdir(p):
        idpath = os.path.join(p, ids)
        sty = random.choice([0, 1, 2, 3])
        nidpath = os.path.join(p, ids + f"_{sty}")
        os.rename(idpath, nidpath)   

if __name__ == '__main__':

    args = parse_args()
    num_case = 4
    num_percase = args.num_percase
    num_ids = num_case * num_percase
    num_sams = args.num_sams
    
    crop_path = args.crop_path
    pv_path = args.pv_path
    aug_path = args.aug_path
    palmprint_path = args.palmprint_path
    erapalmprint_path = args.erapalmprint_path
    
    cvt(crop_path)
    # Data augmentation: Local absence of patterns.
    era_aug(palmprint_path, erapalmprint_path, sams=num_sams)
    # Blend palmprint lines and palm vein patterns.
    blend_folder(vein_path=crop_path, palmprint_path=erapalmprint_path, pv_path=pv_path)
    # Data augmentation: rotation, distortion, etc.
    aug(pv_path, aug_path)
    # Add style lable to ID name
    ChangeIDName(aug_path)

    
