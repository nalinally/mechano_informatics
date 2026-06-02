import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import os

from hopfield_network import HopfieldNetwork
import img_resize_binarize
from common import *

def read_data_from_img_file(file_path):
    # print(f"dbg0 {file_path}")
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
    data_reshaped = np.array(data).reshape(img_size[0], img_size[1])
    img = [[[0, 0, 0] if x == 1 else [255, 255, 255] for x in col] for col in data_reshaped]
    img = np.array(img)
    img = img.astype(np.uint8)
    return img
    
def add_noise(data, ratio):
    rng = np.random.default_rng()
    noised_id = []
    data_noised = data.copy()
    n_random = int(len(data_noised) * ratio)
    while(True):
        id = rng.integers(len(data_noised))
        if not (id in noised_id):
            data_noised[id] = -data_noised[id]
            noised_id.append(id)
        if len(noised_id) >= n_random:
            break
    return data_noised
    
def similarity(data1, data2):
    return np.dot(data1, data2) / len(data1)

def recall_until_stabilized(net):
    pre_x = net.getX()
    while(True):
        x = net.recall(1000)
        if x == pre_x:
            break
    return net

def recall_test(net, imgs):
    id = 0
    noise_ratio = 0.2
    step = 1
    steps = 0
    net.learn([imgs[id]])
    net.set(add_noise(imgs[id], noise_ratio))
    os.makedirs(recall_test_fig_dir+f"{id}", exist_ok=True)
    for _ in range(100):
        img = data_to_img(net.recall(step))
        steps += step
        # print(recall_test_fig_dir+f"{id}/{steps}.png")
        cv2.imwrite(recall_test_fig_dir+f"{id}/{steps}.png", img)

def simple_performance_test(net, imgs):
    id = [0, 1, 2, 3, 4, 5]
    noise_ratio = 0.2
    n_test = 1000
    similarities = []
    torf = []
    net.learn([imgs[id_] for id_ in id])
    for id_ in id:
        print(f"\ntest pattern {id_}.\n")
        for _ in range(n_test):
            net.set(add_noise(imgs[id_], noise_ratio))
            # cv2.imshow("before", cv2.hconcat([data_to_img(imgs[id_]), data_to_img(net.getX())]))
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            recall_until_stabilized(net)
            # cv2.imshow("after", cv2.hconcat([data_to_img(imgs[id_]), data_to_img(net.getX())]))
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            similarity_ = similarity(imgs[id_], net.getX())
            # print(f"similarity: {similarity_}")
            similarities.append(similarity_)
            torf.append(1 if similarity_ == 1 else 0)
    print(f"average of similarities: {np.mean(similarities)}")
    print(f"true ratio: {np.mean(torf)}")
    
def noise_performance_test(net, imgs):
    id = [0, 1, 2, 3]
    csv_file = 'noise_performance_test_4pattern.csv'
    noise_ratio = np.linspace(0, 1, 41)
    n_test = 300
    similarities_list = []
    torf_list = []
    net.learn([imgs[id_] for id_ in id])
    for noise_ratio_ in noise_ratio:
        print(f"\ntest noise {noise_ratio_}.\n")
        similarities = []
        torf = []
        for id_ in id:
            print(f"\ntest pattern {id_}.\n")
            for _ in range(n_test):
                net.set(add_noise(imgs[id_], noise_ratio_))
                # cv2.imshow("before", cv2.hconcat([data_to_img(imgs[id_]), data_to_img(net.getX())]))
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                recall_until_stabilized(net)
                # cv2.imshow("after", cv2.hconcat([data_to_img(imgs[id_]), data_to_img(net.getX())]))
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                similarity_ = similarity(imgs[id_], net.getX())
                # print(f"similarity: {similarity_}")
                similarities.append(similarity_)
                torf.append(1 if similarity_ == 1 else 0)
        similarities_list.append(similarities)
        torf_list.append(torf)
        print(f"average of similarities: {np.mean(similarities)}")
        print(f"true ratio: {np.mean(torf)}")
    with open(csv_dir + csv_file, 'w') as f:
        title_row = np.hstack([["noise_ratio"], [f"sm[{i}]" for i in range(len(torf_list[0]))], [f"tf[{i}]" for i in range(len(torf_list[0]))]])
        write_list = np.hstack([np.array([noise_ratio]).T, similarities_list, torf_list])
        writer = csv.writer(f)
        writer.writerow(title_row)
        writer.writerows(write_list)
    plt.plot(noise_ratio, np.mean(similarities_list, axis=1))
    plt.show()
    
def csv_write_test():
    csv_file = 'test.csv'
    noise_ratio = [0, 0.5, 1]
    similarities_list = [[1, 0.9, 1, 1, 0.89], [0, 0.1, -0.1, 0, 0.01], [-1, -1, -0.85, -1, -0.9]]
    torf_list = [[1, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    with open(csv_dir + csv_file, 'w') as f:
        title_row = np.hstack([["noise_ratio"], [f"sm[{i}]" for i in range(len(torf_list[0]))], [f"tf[{i}]" for i in range(len(torf_list[0]))]])
        write_list = np.hstack([np.array([noise_ratio]).T, similarities_list, torf_list])
        writer = csv.writer(f)
        writer.writerow(title_row)
        writer.writerows(write_list)
    
def main():
    net = HopfieldNetwork(img_size[0] * img_size[1])
    imgs = read_data_from_dir(processed_img_dir)
    # recall_test(net, imgs)
    simple_performance_test(net, imgs)
    # noise_performance_test(net, imgs)
    # csv_write_test()
    
if __name__=="__main__":
    main()
