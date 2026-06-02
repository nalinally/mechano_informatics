import numpy as np
import matplotlib.pyplot as plt
import csv

from common import *

noise_performance_test_result_file = "noise_performance_test_4pattern.csv"

noise_ratio = None
similarities_list = None
torf_list = None

simple_test_pattern_num = [0, 1, 2, 3, 4, 5]
simple_test_similarities_list = [1.0, 0.9908, 0.9641066666666666, 0.8799599999999999, 0.7773599999999999, 0.6653733333333333]
simple_test_torf_list = [1.0, 0.9845, 0.9163333333333333, 0.64475, 0.4416, 0.19666666666666666]

def read_noise_performance_test_result(csv_file):
    global noise_ratio
    global similarities_list
    global torf_list
    with open(csv_dir + csv_file) as f:
        reader = csv.reader(f)
        data_list = [row for row in reader]
        data_list = np.array(data_list[1:])
        print(data_list.shape)
        n_trial = int((len(data_list[0]) - 1) / 2)
        print(n_trial)
        noise_ratio = list(map(float, data_list[:, 0]))
        similarities_list = [list(map(float, row)) for row in data_list[:, 1:n_trial + 1]]
        torf_list = [list(map(float, row)) for row in data_list[:, n_trial + 1:]]

def analyze_noise_performance_test_result():
    plt.plot(noise_ratio, np.mean(similarities_list, axis=1), label="similarity")
    plt.plot(noise_ratio, np.mean(torf_list, axis=1), label="accuracy")
    plt.title("noise and performance(kind of fig:4)")
    plt.xlabel("noise")
    plt.ylabel("performance")
    plt.legend()
    plt.show()
    
def analyze_simple_performance_test_result():
    plt.plot(simple_test_pattern_num, simple_test_similarities_list, label="similarity")
    plt.plot(simple_test_pattern_num, simple_test_torf_list, label="accuracy")
    plt.title("num of patterns and performance")
    plt.xlabel("num of patterns")
    plt.ylabel("performance")
    plt.legend()
    plt.show()
    
def main():
    read_noise_performance_test_result(noise_performance_test_result_file)
    # analyze_noise_performance_test_result()
    analyze_simple_performance_test_result()

if __name__=="__main__":
    main()
    