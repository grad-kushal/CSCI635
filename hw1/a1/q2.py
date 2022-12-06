import os

import numpy as np
import sklearn
import torch as torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


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


class BinaryLogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(BinaryLogisticRegressionModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_features, output_features)

    def forward(self, xx):
        y_predicted = torch.sigmoid(self.layer1(xx))
        return y_predicted


if __name__ == '__main__':
    files = ["/Frogs.csv", "/Frogs-subsample.csv"]
    path = os.path.abspath(os.getcwd())

    for file in files:
        data = read_data(path + file)
        type_1 = data['HylaMinuta\n']
        type_2 = data['HypsiboasCinerascens\n']

        type_1_x = []
        type_1_y = []
        type_2_x = []
        type_2_y = []

        x = []
        y = []

        for value in type_1:
            splits = value.split(',')
            x.append([float(splits[0]), float(splits[1])])
            y.append(0)
            type_1_x.append(float(splits[0]))
            type_1_y.append(float(splits[1]))
        for value in type_2:
            splits = value.split(',')
            x.append([float(splits[0]), float(splits[1])])
            y.append(1)
            type_2_x.append(float(splits[0]))
            type_2_y.append(float(splits[1]))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        # scaler = sklearn.preprocessing.StandardScaler()
        # x_train = scaler.fit_transform(x_train)
        # x_test = scaler.fit_transform(x_test)
        # y_train = scaler.fit_transform([y_train])
        # y_test = scaler.fit_transform([y_test])
        #
        # x_train = torch.from_numpy(x_train.astype(np.float32))
        # x_test = torch.from_numpy(x_test.astype(np.float32))
        # y_train = torch.from_numpy(y_train.astype(np.float32))
        # y_test = torch.from_numpy(y_test.astype(np.float32))
        #
        # y_train = y_train.view(625, 1)
        # y_test = y_test.view(157, 1)

        model = BinaryLogisticRegressionModel(2, 1)
        loss_type = torch.nn.BCELoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
        y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

        number_of_epochs = 100000
        loss_vals = []
        for epoch in range(number_of_epochs):
            labels = y_train
            attrs = x_train
            opt.zero_grad()
            y_prediction = model(x_train)
            loss = loss_type(torch.squeeze(y_prediction), labels)
            loss.backward()
            opt.step()
            if (epoch + 1) % 1000 == 0:
                with torch.no_grad():
                    hits = 0
                    total = 0
                    total += y_train.size(0)
                    hits += np.sum(torch.squeeze(y_prediction).round().detach().numpy() == y_train.detach().numpy())
                    accuracy = 100 * hits / total
                    loss_vals.append(loss.item())

                    print("Loss: " + str(loss.item()) + ". \tAccuracy: " + str(accuracy));

        parm = {}
        b = []
        for name, param in model.named_parameters():
            parm[name] = param.detach().numpy()

        print(parm)

        w = parm['layer1.weight'][0]
        b = parm['layer1.bias'][0]
        type_1_scatter_plot = plt.scatter(type_1_x, type_1_y, c='green')
        type_2_scatter_plot = plt.scatter(type_2_x, type_2_y, c='red')

        u = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 2)
        plt.plot(u, (0.5 - b - w[0] * u) / w[1])
        plt.title("Scatter plot for File: " + file)
        plt.xlabel('MFCCs_10')
        plt.ylabel('MFCCs_17')
        plt.legend((type_1_scatter_plot, type_2_scatter_plot), ('HylaMinuta', 'HypsiboasCinerascens'))
        plt.show()
