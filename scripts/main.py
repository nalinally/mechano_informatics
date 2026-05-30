import numpy as np
import matplotlib.pyplot as plt
import cv2

from hopfield_network import HopfieldNetwork
import img_resize_binarize
from common import *

def read_data_from_img_file(file_path):
    print(f"dbg0 {file_path}")
    img = cv2.imread(file_path)
    img_res = img.reshape(-1)[::3]
    img_bin = [1 if x >= 128 else -1 for x in img_res]
    return img_bin
    
def read_data_from_dir(dir_path):
    imgs = []
    img_files = img_resize_binarize.get_image_files(dir_path)
    for img_file in img_files:
        imgs.append(read_data_from_img_file(dir_path + img_file))
    return imgs

def data_to_img(data):
    data.reshape(img_size[0], img_size[1])
    
def main():
    net = HopfieldNetwork(img_size[0] * img_size[1])
    imgs = read_data_from_dir(processed_img_dir)
    net.learn(imgs)
    
if __name__=="__main__":
    main()
