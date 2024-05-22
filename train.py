import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau

from utils import get_network, CustomLRScheduler
# from tinyImagenet import *
from myDataLoader import *
from tqdm import tqdm
from sklearn import metrics
from torch.autograd import Variable
from test import test


validation_loss = []

Accuracy_list = []
Top1_error = []
Top5_error = []


def train_mix(epoch, model_name):
    def mixup_data(x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).to(device)
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    for epoch in range(epoch):
        model.train()

        train_bar = tqdm(train_dataloader)
        all_len = 0
        train_loss = 0
        total = 0
        correct = 0
        for index, (image, label) in enumerate(train_bar):
            image, label = image.to(device), label.to(device)
            # mixup
            inputs, targets_a, targets_b, lam = mixup_data(image, label,
                                                           alpha=0.2, use_cuda=True)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                          targets_a, targets_b))

            # mix loss
            optimizer.zero_grad()
            y = model(inputs)
            y = y.squeeze(dim=1)
            pred = torch.argmax(y, dim=1)
            loss = mixup_criterion(loss_function, y, targets_a, targets_b, lam)
            # running_acc += (pred == labels.data).sum()
            train_loss += loss.item()
            _, predicted = torch.max(y.data, 1)
            total += label.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            loss.backward()
            optimizer.step()
            all_len += len(image)
            # train_bar.desc = "[{}/{}] | Loss: {:.8f} | Acc: {:.8f} ({}/{})"\
            #     .format(index, len(train_dataloader), train_loss/(index+1), 100.*correct/total, correct, total)
            train_bar.desc = "epoch [{}/{}] | Loss: {:.8f} | Acc: {:.8f} ({}/{})" \
                .format(epoch + 1, epochs, train_loss / (index + 1), 100. * correct / total, correct, total)

        print("train: loss:{:.8f} acc:{:.8f}%".format(train_loss / index, 100. * correct / total))

        scheduler.step()
        if epoch >= 85:
            with torch.no_grad():
                correct_1 = 0.0
                correct_5 = 0.0
                all_data = 0
                model.eval()

                valid_bar = tqdm(valid_dataloader)
                for n_iter, (image, label) in enumerate(valid_bar):
                    image = image.to(device)
                    label = label.to(device)

                    output = model(image)
                    _, pred = output.topk(5, 1, largest=True, sorted=True)

                    label = label.view(label.size(0), -1).expand_as(pred)
                    correct = pred.eq(label).float()

                    # compute top 5
                    correct_5 += correct[:, :5].sum()

                    # compute top1
                    correct_1 += correct[:, :1].sum()
                    all_data += len(label)
                print("Top 1 err: ", 1 - (correct_1 / all_data))
                print("Top 5 err: ", 1 - (correct_5 / all_data))
    torch.save(model.state_dict(), 'cifar10-best_model/{}_mix0.2_best_model.pth'.format(model_name))


def train(epoch, model_name):
    for epoch in range(epoch):

        model.train()
        epoch_acc = 0.0
        train_bar = tqdm(train_dataloader)
        all_len = 0
        epoch_loss = 0.0
        for index, (image, label) in enumerate(train_bar):
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            y = model(image)
            y = y.squeeze(dim=1)
            pred = torch.argmax(y, dim=1)
            loss = loss_function(y, label)
            loss.backward()
            optimizer.step()
            all_len += len(image)
            epoch_loss += loss.item()
            epoch_acc += (pred == label.data).sum().item()
            train_bar.desc = "train: epoch [{}/{}] loss:{:.8f} acc:{}".format(epoch + 1,
                                                                              epochs,
                                                                              loss,
                                                                              epoch_acc,
                                                                              )
        epoch_loss = epoch_loss / all_len
        epoch_acc = epoch_acc / all_len
        print("train: loss:{:.8f} acc:{:.8f}%".format(epoch_loss, epoch_acc * 100))
        # 更新学习率调度器
        scheduler.step()
        if epoch >= 180:
            with torch.no_grad():
                correct_1 = 0.0
                correct_5 = 0.0
                all_data = 0
                model.eval()

                valid_bar = tqdm(valid_dataloader)
                for n_iter, (image, label) in enumerate(valid_bar):
                    image = image.to(device)
                    label = label.to(device)

                    output = model(image)
                    _, pred = output.topk(5, 1, largest=True, sorted=True)

                    label = label.view(label.size(0), -1).expand_as(pred)
                    correct = pred.eq(label).float()

                    # compute top 5
                    correct_5 += correct[:, :5].sum()

                    # compute top1
                    correct_1 += correct[:, :1].sum()
                    all_data += len(label)
                print("Top 1 err: ", 1 - (correct_1 / all_data))
                print("Top 5 err: ", 1 - (correct_5 / all_data))
    torch.save(model.state_dict(), 'cifar10-best_model/{}_edge32_best_model.pth'.format(model_name))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')

    # args = parser.parse_args()
    # args = 'resnext26_32x4d'
    import torchvision.models as models

    MILESTONES = [60, 120, 160]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    warmup_epochs = 5

    warmup_lr_init = 0.1
    warmup_lr_end = 0.4
    epochs = 200

    lr = 0.2
    loss_function = nn.CrossEntropyLoss()
    # preActRes18   resnet34   resnet50   resnext26_32x4d
    model_list = ['preActRes18']
    for model_name in model_list:
        if model_name == 'preActRes18':
            args = 'preactresnet18'
            model = get_network(args, num_classes=10)
            model = model.to(device)
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = CustomLRScheduler(optimizer)
            train(epochs, model_name)
            # train_mix(epochs, model_name)
            print()
        if model_name == 'resnet34':
            args = models.resnet34(pretrained=False)
            model = args
            model.fc = torch.nn.Linear(in_features=512, out_features=10)
            model.to(device)
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = CustomLRScheduler(optimizer)
            # train(epochs, model_name)
            train_mix(epochs, model_name)
            print()
        elif model_name == 'resnet50':
            args = models.resnet50(pretrained=False)
            model = args
            model.fc = torch.nn.Linear(in_features=2048, out_features=10)
            model.to(device)
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = CustomLRScheduler(optimizer)
            # train(epochs, model_name)
            train_mix(epochs, model_name)
            print()
        elif model_name == 'resnext50_32x4d':
            args = 'resnext50_32x4d'
            model = get_network(args, num_classes=10)
            model = model.to(device)
            # model = nn.DataParallel(model, device_ids=[0, 1, 2])
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = CustomLRScheduler(optimizer)
            # train(epochs, model_name)
            train_mix(epochs, model_name)
            print()
