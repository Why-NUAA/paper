import os
import scipy.io as scio
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def statistics(y_true, pre):
    acc, auc, sen, spe = 0.0, 0.0, 0.0, 0.0
    try:
        ACC = accuracy_score(y_true, pre)
        AUC = roc_auc_score(y_true, pre)
        TP = torch.sum(y_true & pre)
        TN = len(y_true) - torch.sum(y_true | pre)
        true_sum = torch.sum(y_true)
        neg_sum = len(y_true) - true_sum
        SEN = TP / true_sum
        SPE = TN / neg_sum

        acc += ACC
        sen += SEN.cpu().numpy()
        spe += SPE.cpu().numpy()
        auc += AUC

    except ValueError as ve:
        print(ve)
        pass

    return acc, sen, spe, auc


def pick_data(data, pick):
    picked_data = []
    for i in range(len(data)):
        if data[i][-1] in pick:
            picked_data.append(data[i])
    return np.array(picked_data)


def change_label(labels):
    if 0 not in labels:
        for i, label in enumerate(labels):
            labels[i] -= 1
    else:
        for i, label in enumerate(labels):
            if labels[i] != 0:
                labels[i] = 1
    return labels


def load_dataset(file_path, file_name, pick=[1, 2]):
    epilepsy = scio.loadmat(os.path.join(file_path, file_name[0]))
    dti = scio.loadmat(os.path.join(file_path, file_name[1]))['G'].transpose(2, 0, 1)
    mri = epilepsy["data"]
    gnd = epilepsy['gnd'][0, :]
    label = gnd.reshape((gnd.shape[0], -1))

    b, h, w = mri.shape
    _, n, m = dti.shape
    scaler = StandardScaler()
    mri = mri.reshape(b, -1)
    dti = dti.reshape(b, -1)
    mri = scaler.fit_transform(mri)
    dti = scaler.fit_transform(dti)
    data = np.concatenate((mri, dti, label), axis=1)

    data = pick_data(data, pick)
    mri = data[:, :h * w]
    dti = data[:, h * w: -1]

    mri = mri.reshape(-1, h, w)
    dti = dti.reshape(-1, n, m)
    label = data[:, -1]
    return np.concatenate((mri, dti), axis=2), label


def split_dataset(data, label, ki, K, valid_ratio=0.15):
    test = []
    index = [x for x in range(K)]
    test_index = index.pop(ki)
    classes = list(set(label))
    num_classes = len(classes)
    class_index = [[] for i in range(num_classes)]
    for i, x in enumerate(label):
        class_index[classes.index(x)].append(i)

    for x in class_index:
        np.random.shuffle(x)
    sample_index = [x for x in range(0, label.shape[0])]

    every_k_len = []
    for x in class_index:
        if len(x) / K - len(x) // K < 0.5:
            every_k_len.append(len(x) // K)
        else:
            every_k_len.append(len(x) // K + 1)

    for i, x in enumerate(class_index):
        if test_index != K - 1:
            test.extend(x[every_k_len[i] * test_index: every_k_len[i] * (test_index + 1)])
        else:
            test.extend(x[every_k_len[i] * test_index:])

    test_flag = torch.tensor([True if x in test else False for x in sample_index])

    train_index = list(set(sample_index) - set(test))
    extract_valid = np.random.randint(len(train_index), size=int(len(train_index) * valid_ratio))
    valid_index = [index if i in extract_valid else -1 for i, index in enumerate(train_index)]
    valid_index = list(set(valid_index) - {-1})
    valid_flag = torch.tensor([True if x in valid_index else False for x in sample_index])
    train_index = list(set(train_index) - set(valid_index))
    train_flag = torch.tensor([True if x in train_index else False for x in sample_index])

    b, h, w = data.shape
    data = data.reshape(b, -1)
    data_label = np.concatenate((data, label.reshape(b, -1)), axis=1)
    trainset = data_label[train_flag]
    testset = data_label[test_flag]
    validset = data_label[valid_flag]
    np.random.shuffle(trainset)
    np.random.shuffle(testset)
    np.random.shuffle(validset)

    return trainset, validset, testset