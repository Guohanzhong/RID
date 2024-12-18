import os
import json
import numpy
import numpy as np
from PIL import Image
import logging
import os.path as osp
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from imageio import imread
from utils import (preprocess, 
                    prepare_mask_and_masked_image,
                    prepare_nonmask_and_nonmasked_image,prepare_image )

class IMAGE(Dataset):

    def __init__(self, path,resize=False,image_size=512):
        self.path = path
        #print(os.listdir(self.path))
        self.folders = [osp.join(self.path, d) for d in os.listdir(self.path)]
        self.resize = resize
        self.image_size = image_size
        self.images = []
        #logging.info(f'the list of image is {self.folders}')
        for i, folder in enumerate(self.folders):
            #im_path = [osp.join(folder, im) for im in os.listdir(folder)]
            self.images.append(folder)
        logging.info(f'the list of image is {self.images}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        init_image = Image.open(path).convert('RGB')
        init_image = init_image.resize((self.resize,self.resize))
        init_image, cur_mask, cur_masked_image = prepare_nonmask_and_nonmasked_image(init_image,self.image_size)
        cur_mask = cur_mask.half()
        cur_masked_image = cur_masked_image.half()
        #init_image = prepare_image(init_image)
        return init_image.squeeze(0),cur_mask.squeeze(0),cur_masked_image.squeeze(0)

if __name__ == '__main__':
    topdir = "./image/Face"
    output_dir = "./image/Face_sorted"
    os.makedirs(output_dir,exist_ok=True)
    img_dir_list = sorted(os.listdir(topdir))
    final_json = {}
    ans = 0
    for name_ele in img_dir_list:
        img_name_file = os.path.join(topdir,name_ele)
        for set_ele in sorted(os.listdir(img_name_file)):
            img_set_file = os.path.join(img_name_file,set_ele)
            if 'DS_Store' in img_set_file:
                continue
            final_image_list = sorted(os.listdir(img_set_file))
            for img in final_image_list:
                img_path = os.path.join(img_set_file,img)
                #if ans == 0:
                final_json[ans] = img_path
                img_temp = Image.open(img_path)
                output_img_path = os.path.join(output_dir,str(ans)+'.png')
                img_temp.save(output_img_path)
                ans += 1
    with open('./image/face_data.json','w') as file:
        for key, value in final_json.items():  
            line = json.dumps({key: value})  
            file.write(line + '\n')  
