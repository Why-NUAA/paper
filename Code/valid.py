import torch
from utils import statistics


def valid(train_loader, valid_loader, model, num_classes):
    predict, y_true = None, None
    trained_embedding, trained_label = None, None

    model.eval()
    with torch.no_grad():
        for i, (inputs, label) in enumerate(train_loader):
            inputs = inputs.cuda()
            label = label.cuda()

            train_embedding, _ = model(inputs)
            train_label = label.view(-1)
            if i == 0:
                trained_embedding = train_embedding
                trained_label = train_label
            else:
                trained_embedding = torch.vstack((trained_embedding, train_embedding))
                trained_label = torch.hstack((trained_label, train_label))

        for i, (inputs, label) in enumerate(valid_loader):
            inputs = inputs.cuda()
            label = label.cuda()

            embedding, _ = model(inputs)

            for j, emb in enumerate(embedding):
                dist = torch.sum((emb - trained_embedding) ** 2, dim=-1) ** 0.5
                top_k = torch.argsort(dist)[:5]
                count = [0 for i in range(num_classes)]
                for k in trained_label[top_k]:
                    count[k] += 1
                emb_label = torch.argmax(torch.tensor(count))

                if i == 0 and j == 0:
                    predict = emb_label
                else:
                    predict = torch.hstack((predict, emb_label))

            label = label.view(-1)

            if i == 0:
                pre = predict
                y_true = label
            else:
                pre = torch.hstack((pre, predict))
                y_true = torch.hstack((y_true, label))

        valid_acc, valid_sen, valid_spe, valid_auc = statistics(y_true.cpu(), pre.cpu())

        return valid_acc, valid_sen, valid_spe, valid_auc