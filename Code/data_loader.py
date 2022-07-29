import numpy as np
import torch
import torch.utils.data as Data
from utils import change_label, split_dataset


class GetKfoldLoader(Data.Dataset):
    def __init__(self, data, datashape):
        super(GetKfoldLoader, self).__init__()
        self.data = data[:, :-1].reshape(-1, datashape[1], datashape[2])
        self.label = data[:, -1].astype(np.int)
        self.label = change_label(self.label)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.int64)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.label[item]


def load_data(data, label, batch_size, num_workers, ki=0, fold=10, valid_ratio=0.15):
    trainset, validset, testset = split_dataset(data, label, ki, fold, valid_ratio)
    evalset = np.concatenate((trainset, validset), axis=0)

    train_loader = GetKfoldLoader(data=trainset, datashape=data.shape)
    valid_loader = GetKfoldLoader(data=validset, datashape=data.shape)
    test_loader = GetKfoldLoader(data=testset, datashape=data.shape)
    eval_loader = GetKfoldLoader(data=evalset, datashape=data.shape)

    train_dataloader = Data.DataLoader(
        dataset=train_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )
    valid_dataloader = Data.DataLoader(
        dataset=valid_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    test_dataloader = Data.DataLoader(
        dataset=test_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    eval_dataloader = Data.DataLoader(
        dataset=eval_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    return train_dataloader, valid_dataloader, test_dataloader, eval_dataloader