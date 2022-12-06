import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def read_data(filename):
    data_map = dict()
    with open(filename, "r") as file_object:  # open file
        for line in file_object:
            split_line = line.split(',')
            if split_line[2] != 'Species\n':
                if not split_line[2] in data_map:
                    data_map[split_line[2]] = []
                data_map.get(split_line[2]).append(str(split_line[0]) + ',' + str(split_line[1]))
    return data_map


if __name__ == '__main__':
    files = ["/Frogs.csv", "/Frogs-subsample.csv"]
    path = os.path.abspath(os.getcwd())

    print('matplotlib: {}'.format(matplotlib.__version__))

    for file in files:
        data = read_data(path + file)
        data_type_1 = data['HylaMinuta\n']
        data_type_2 = data['HypsiboasCinerascens\n']

        type_1_x = []
        type_1_y = []
        type_2_x = []
        type_2_y = []

        for value in data_type_1:
            type_1_x.append(float(value.split(',')[0]))
            type_1_y.append(float(value.split(',')[1]))
        for value in data_type_2:
            type_2_x.append(float(value.split(',')[0]))
            type_2_y.append(float(value.split(',')[1]))

    ################################## Scatter Plot #########################################

        type_1_scatter_plot = plt.scatter(type_1_x, type_1_y, c='green')
        type_2_scatter_plot = plt.scatter(type_2_x, type_2_y, c='red')
        plt.title("Scatter plot for File: " + file)
        plt.xlabel('MFCCs_10')
        plt.ylabel('MFCCs_17')
        plt.legend((type_1_scatter_plot, type_2_scatter_plot), ('HylaMinuta', 'HypsiboasCinerascens'))
        plt.show()

    ################################## Histograms #########################################

        plt.hist(type_1_x, color='yellow', edgecolor='black')
        plt.title("File: " + file + "\nMFCCs_10 Histogram for class: HylaMinuta")
        plt.xlabel('MFCCs_10')
        plt.ylabel('Frequency')
        plt.show()

        plt.hist(type_1_y, color='blue', edgecolor='black')
        plt.title("File: " + file + "\nMFCCs_17 Histogram for class: HylaMinuta")
        plt.xlabel('MFCCs_17')
        plt.ylabel('Frequency')
        plt.show()

        plt.hist(type_2_x, color='yellow', edgecolor='black')
        plt.title("File: " + file + "\nMFCCs_10 Histogram for class: HypsiboasCinerascens")
        plt.xlabel('MFCCs_10')
        plt.ylabel('Frequency')
        plt.show()

        plt.hist(type_2_y, color='blue', edgecolor='black')
        plt.title("File: " + file + "\nMFCCs_17 Histogram for class: HypsiboasCinerascens")
        plt.xlabel('MFCCs_17')
        plt.ylabel('Frequency')
        plt.show()

    ################################## Line Graphs #########################################

        sorted_type_1_x = list(type_1_x)
        sorted_type_1_x.sort()
        sorted_type_1_y = list(type_1_y)
        sorted_type_1_y.sort()
        plt.plot(sorted_type_1_x, sorted_type_1_y)
        plt.xlabel('MFCCs_10')
        plt.ylabel('MFCCs_17')
        plt.title("Line Graph for file: " + file + "and class: HylaMinuta")
        plt.show()

        sorted_type_2_x = list(type_2_x)
        sorted_type_2_x.sort()
        sorted_type_2_y = list(type_2_y)
        sorted_type_2_y.sort()
        plt.plot(sorted_type_2_x, sorted_type_2_y)
        plt.xlabel('MFCCs_10')
        plt.ylabel('MFCCs_17')
        plt.title("Line Graph for file: " + file + " and class: HypsiboasCinerascens")
        plt.show()

    ################################## Box Plots #########################################
        plt.boxplot([type_1_x, type_1_y, type_2_x, type_2_y], labels=['HylaMinuta-MFCCs_10', 'HylaMinuta-MFCCs_17', 'HypsiboasCinerascens-MFCCs_10', 'HypsiboasCinerascens-MFCCs_17'])
        plt.title("BoxPlots for file: " + file)
        plt.show()

    ################################## Errors #########################################

        labels = ["HM MFCCs_10", "HM MFCCs_17", "HC MFCCs_10", "HC MFCCs_17"]
        x_pos = np.arange(len(labels))
        CTEs = [np.mean(type_1_x), np.mean(type_1_y), np.mean(type_2_x), np.mean(type_2_y)]
        error = [np.std(type_1_x), np.std(type_1_y), np.std(type_2_x), np.std(type_2_y)]

        fig, ax = plt.subplots()
        ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.yaxis.grid(True)
        plt.tight_layout()
        plt.show()

    ################################## Descriptive #########################################
        print('File: ' + file + ', MFCCs_10 Mean: ' + str(np.mean(type_1_x + type_2_x)))
        print('File: ' + file + ', MFCCs_17 Mean: ' + str(np.mean(type_1_y + type_2_y)))

        cov_list = []
        cov_list.append(type_1_x + type_2_x)
        cov_list.append(type_1_y + type_2_y)
        print('File: ' + file + ', Covariance Matrix: \n' + str(np.cov(cov_list)))

        print('File: ' + file + ', MFCCs_10 Standard Deviation: ' + str(np.std(type_1_x + type_2_x)))
        print('File: ' + file + ', MFCCs_17 Standard Deviation: ' + str(np.std(type_1_y + type_2_y)))
