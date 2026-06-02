import numpy as np
import matplotlib.pyplot as plt
import csv

from common import *

def read_noise_performance_test_result(csv_file):
    with open(csv_dir + csv_file) as f:
        reader = csv.reader(f)
        data_list = [row for row in reader]
        data_list = np.array(data_list[1:])
        print(data_list.shape)
        n_trial = int((len(data_list[0]) - 1) / 2)
        print(n_trial)
        noise_ratio = data_list[:, 0]
        similarities_list = data_list[1:n_trial, :]
        torf_list = data_list[n_trial + 1:, :]
        print(noise_ratio.shape)
        print(noise_ratio)
        print(similarities_list.shape)
        print(torf_list.shape)
    
def main():
    read_noise_performance_test_result("noise_performance_test_2pattern.csv")

if __name__=="__main__":
    main()
    