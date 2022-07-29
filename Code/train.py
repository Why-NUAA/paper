import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_dataset
from data_loader import load_data
from loss_function import MyTriplet_loss
from Triplet_OringinT.Final_Code.net import net
from test import test
from valid import valid

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, criterions, optimizer, train_loader, valid_loader, fold, epochs, num_classes):
    train_size = len(train_loader)
    train_loss, valid_loss = [], []

    current_test_acc, best_valid_epoch, valid_epoch_loss = 0, 0, 0
    best_acc = 0
    best_model = None

    for epoch in range(epochs):
        start = time.time()
        model.train()
        epoch_loss, CE_loss, triplet_loss = 0, 0, 0

        # train
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda()
            label = labels.view(-1).cuda()
            embedding, x = model(inputs)
            predict = model.frozen_forward(inputs)

            loss1 = criterions[0](predict, label)
            loss2 = criterions[1](embedding, label)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            CE_loss += loss1.item()
            triplet_loss += loss2.item()

        # validation
        valid_acc, valid_sen, valid_spe, valid_auc = valid(train_loader, valid_loader, model, num_classes)

        if valid_acc >= best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, "best_epoch_fold{}.pkl".format(fold))

        epoch_loss = epoch_loss / train_size
        train_loss.append(epoch_loss)
        valid_loss.append(valid_epoch_loss)
        CE_loss = CE_loss / train_size
        triplet_loss = triplet_loss / train_size

        end = time.time() - start
        print("\033[31;1m < F{} {:.0f}% {}/{} {:.3f}s >\033[0m "
              .format(fold, (epoch + 1) / epochs * 100, epoch + 1, epochs, end), end="")
        print('train_loss =', '\033[32;1m{:.5f} \033[0m'.format(epoch_loss), 'ce_loss =', '{:.5f}'.format(CE_loss),
              'triplet_loss =', '{:.5f} '.format(triplet_loss), end="")
        print('valid_acc=\033[31;1m {:.4f} \033[0m'.format(valid_acc * 100))


if __name__ == '__main__':
    file_path = r"../../Data/Epilepsy"
    file_name = [r"X_data_gnd", r"G_all"]

    # Parameters
    timepoints, rois = 240, 90
    dim, depth, heads = 256, 1, 1
    dropout = 0.5
    batch_size = 128
    epochs = 200
    num_classes = 2

    # tasks
    pick = [0, 1]
    # pick = [0, 2]
    # pick = [1, 2]
    # pick = [0, 1, 2]

    # get data
    data, label = load_dataset(file_path, file_name, pick=pick)
    valid_ratio = 0.2
    alpha, beta = 0.5, 0.5

    # k-fold validation
    predict_acc, predict_auc, predict_sen, predict_spe = [], [], [], []

    K = 10
    for ki in range(K):

        train_loader, valid_loader, test_loader, eval_loader = load_data(
            data, label, batch_size=batch_size, num_workers=0,
            ki=ki, fold=K, valid_ratio=valid_ratio)

        model = net(rois, timepoints, num_classes, depth, heads, dropout).cuda()

        criterion1 = nn.CrossEntropyLoss().cuda()
        criterion2 = MyTriplet_loss(margin=0.8, loss_weight=alpha).cuda()
        criterions = [criterion1, criterion2]

        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-5)

        train(model, criterions, optimizer, train_loader, valid_loader, ki + 1, num_classes=num_classes, epochs=epochs)

        test_model = torch.load("best_epoch_fold{}.pkl".format(ki + 1))
        acc, SEN, SPE, auc = test(test_loader, eval_loader, test_model, num_classes)

        predict_acc.append(acc)
        predict_auc.append(auc)
        predict_sen.append(SEN)
        predict_spe.append(SPE)
