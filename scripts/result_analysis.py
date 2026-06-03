import numpy as np
import matplotlib.pyplot as plt
import csv

from common import *

npt_result_file = "noise_performance_test_4pattern.csv"
npt_noise_ratio = None
npt_similarities_list = None
npt_torf_list = None

st_pattern_num = [0, 1, 2, 3, 4, 5]
st_similarities_list = [1.0, 0.9908, 0.9641066666666666, 0.8799599999999999, 0.7773599999999999, 0.6653733333333333]
st_torf_list = [1.0, 0.9845, 0.9163333333333333, 0.64475, 0.4416, 0.19666666666666666]

spt_result_file = "similarity_performance_test_4pattern_300tests.csv"
spt_sim_l1s = None
spt_sim_l2s = None
spt_l1_l1s = None
spt_l1_l2s = None
spt_l2_l1s = None
spt_l2_l2s = None
spt_similarities_list = None
spt_torf_list = None

def read_noise_performance_test_result(csv_file):
    global npt_noise_ratio
    global npt_similarities_list
    global npt_torf_list
    with open(csv_dir + npt_csv_dir + csv_file) as f:
        reader = csv.reader(f)
        data_list = [row for row in reader]
        data_list = np.array(data_list[1:])
        print(f"[read_npt_result] data shape {data_list.shape}")
        n_trial = int((len(data_list[0]) - 1) / 2)
        print(f"[read_npt_result] num of trial {n_trial}")
        npt_noise_ratio = list(map(float, data_list[:, 0]))
        npt_similarities_list = [list(map(float, row)) for row in data_list[:, 1:n_trial + 1]]
        npt_torf_list = [list(map(float, row)) for row in data_list[:, n_trial + 1:]]

def read_similarity_performance_test_result(csv_file):
    global spt_sim_l1s
    global spt_sim_l2s
    global spt_l1_l1s
    global spt_l1_l2s
    global spt_l2_l1s
    global spt_l2_l2s
    global spt_similarities_list
    global spt_torf_list
    with open(csv_dir + spt_csv_dir + csv_file) as f:
        reader = csv.reader(f)
        data_list = [row for row in reader]
        data_list = np.array(data_list[1:])
        print(f"[read_spt_result] data shape {data_list.shape}")
        n_trial = int((len(data_list[0]) - 6) / 2)
        print(f"[read_spt_result] num of trial {n_trial}")
        spt_sim_l1s = list(map(float, data_list[:, 0]))
        spt_sim_l2s = list(map(float, data_list[:, 1]))
        spt_l1_l1s = list(map(float, data_list[:, 2]))
        spt_l1_l2s = list(map(float, data_list[:, 3]))
        spt_l2_l1s = list(map(float, data_list[:, 4]))
        spt_l2_l2s = list(map(float, data_list[:, 5]))
        spt_similarities_list = [list(map(float, row)) for row in data_list[:, 6:n_trial + 6]]
        spt_torf_list = [list(map(float, row)) for row in data_list[:, n_trial + 6:]]
        # print(np.array(spt_similarities_list).shape)
        # print(np.array(spt_torf_list).shape)

def analyze_noise_performance_test_result():
    plt.plot(npt_noise_ratio, np.mean(npt_similarities_list, axis=1), label="similarity")
    plt.plot(npt_noise_ratio, np.mean(npt_torf_list, axis=1), label="accuracy")
    plt.title("noise vs performance(kind of fig:4)")
    plt.xlabel("noise")
    plt.ylabel("performance")
    plt.legend()
    plt.show()
    
def analyze_similarity_performance_test_result():
    x_datas = [spt_sim_l1s, spt_sim_l2s, spt_l1_l1s, spt_l1_l2s, spt_l2_l1s, spt_l2_l2s]
    x_data_refs = ["l1norm of similarity", "l2norm of similarity", "l1norm of l1norm", "l2norm of l1norm", "l1norm of l2norm", "l2norm of l2norm"]
    similarity_corrcoefs = []
    torf_corrcoefs = []
    for x_data, x_data_ref in zip(x_datas, x_data_refs):
        plt.figure()
        plt.scatter(x_data, np.mean(spt_similarities_list, axis=1), label="similarity")
        plt.scatter(x_data, np.mean(spt_torf_list, axis=1), label="accuracy")
        plt.title(f"{x_data_ref} vs performance(kind of fig:4)")
        plt.xlabel(x_data_ref)
        plt.ylabel("performance")
        plt.legend()
        plt.savefig(fig_dir + spt_fig_dir + f"{x_data_ref}_vs_performance_4patterns.png")
        plt.show(block=False)
        print(f"corrcoef between similarity and {x_data_ref} : {np.corrcoef(x_data, np.mean(spt_similarities_list, axis=1))[0,1]}")
        print(f"corrcoef between accuracy and {x_data_ref} : {np.corrcoef(x_data, np.mean(spt_torf_list, axis=1))[0,1]}")
        similarity_corrcoefs.append(np.corrcoef(x_data, np.mean(spt_similarities_list, axis=1))[0,1])
        torf_corrcoefs.append(np.corrcoef(x_data, np.mean(spt_torf_list, axis=1))[0,1])
    plt.figure()
    plt.title("correlation coefficient between each parameters and simlilarity, accuracy")
    plt.xlabel("correlation coefficient between each parameters and simlilarity")
    plt.ylabel("correlation coefficient between each parameters and accuracy")
    plt.scatter(similarity_corrcoefs, torf_corrcoefs)
    for x, y, ref in zip(similarity_corrcoefs, torf_corrcoefs, x_data_refs):
        plt.text(x, y, ref)
    plt.savefig(fig_dir + spt_fig_dir + "corrcoefs.png")
    plt.show()

def analyze_simple_performance_test_result():
    plt.plot(st_pattern_num, st_similarities_list, label="similarity")
    plt.plot(st_pattern_num, st_torf_list, label="accuracy")
    plt.title("num of patterns and performance")
    plt.xlabel("num of patterns")
    plt.ylabel("performance")
    plt.legend()
    plt.show()

def main():
    read_noise_performance_test_result(npt_result_file)
    read_similarity_performance_test_result(spt_result_file)
    # analyze_noise_performance_test_result()
    # analyze_simple_performance_test_result()
    analyze_similarity_performance_test_result()
    

if __name__=="__main__":
    main()
