import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


from model.TransMIL import TransMIL
import os
import logging
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from utils.plt_metics import plot_confusion_matrix
from tensorboardX import SummaryWriter
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import autocast


class Bag_level_dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        with open(data_path, "r") as f:
            self.data_list = f.readlines()

    def get_bag_feats_v2(self, feats):
        if isinstance(feats, str):
            # if feats is a path, load it
            feats, bag_label = feats.strip().split(',')
            feats = torch.Tensor(np.load(feats))
        else:
            print("error")

        # 打乱
        feats = feats[np.random.permutation(len(feats))]
        bag_label = torch.Tensor([int(bag_label)]).long()

        return bag_label, feats

    def __getitem__(self, index):
        bag_label, feats = self.get_bag_feats_v2(self.data_list[index])
        return feats,bag_label
    def __len__(self):
        return len(self.data_list)


LABELS = ["NILM", "abnormal"]  # 类别的标签名字

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=200, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='Attention', help='Choose b/w attention and gated_attention')
parser.add_argument(
    "--project_name",
    type=str,
    default="TransMIL_adam+lr0.0002_epochs200")  # 项目的名字

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


model = TransMIL()
if args.cuda:
    model.cuda()


project_path = f"/repository7401/Qinjian_data/WSL_classfication_deep_learning/dsmil-wsi/weights交叉熵0.5阈值_qin/Camelyon16/TransMIL/{args.project_name}"

if not os.path.exists(project_path):
    os.makedirs(project_path)
if not os.path.exists(os.path.join(project_path, "results")):
    os.makedirs(os.path.join(project_path, "results"))

if not os.path.exists(os.path.join(project_path, "test_results")):
    os.makedirs(os.path.join(project_path, "test_results"))

test_logging = logging.FileHandler(filename=os.path.join(project_path, "test.txt"), mode='a+', encoding='utf-8')
test_logging.setFormatter(logging.Formatter('%(message)s'))
train_logging = logging.FileHandler(filename=os.path.join(project_path, "train.csv"), mode='a+', encoding='utf-8')
train_logging.setFormatter(logging.Formatter('%(message)s'))

test_logger = logging.Logger("test", level=logging.INFO)
test_logger.addHandler(test_logging)

train_logger = logging.Logger("train", level=logging.INFO)
train_logger.addHandler(train_logging)

writer_train = SummaryWriter(os.path.join(project_path, "train"))
writer_test = SummaryWriter(os.path.join(project_path, "test"))


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    logging.info('\nGPU is ON!')

logging.info('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = data_utils.DataLoader(Bag_level_dataset("/repository7401/Qinjian_data/WSL_classfication_deep_learning/dsmil-wsi/datasets/Camelyon16/remix_processed/train_list.txt"),
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(Bag_level_dataset("/repository7401/Qinjian_data/WSL_classfication_deep_learning/dsmil-wsi/datasets/Camelyon16/remix_processed/test_list.txt"),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)


optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
#optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
# optimizer = optim.SGD(model.parameters(), lr=args.lr)...................
scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=1e-6, warmup_lr_init=0.00001, warmup_t=3)
train_loss_fn = torch.nn.CrossEntropyLoss()

train_logger.info('Epoch train_Loss Train_ACC test_loss test_ACC')


def train(epoch):
    model.train()
    train_loss = 0.
    train_ACC = 0.
    for batch_idx, (data, bag_label) in enumerate(train_loader):
        scheduler.step(epoch)
        # bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        data, bag_label=data.squeeze(0), bag_label.squeeze(0)
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        with autocast():  # 半精度
            Y_prob = model(data)
            loss = train_loss_fn(Y_prob, bag_label)

        _, Y_prob = torch.max(Y_prob, 1)
        ACC = Y_prob.eq(bag_label).cpu().float().mean().item()
        train_loss += loss.item()

        train_ACC += ACC
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_ACC /= len(train_loader)

    # 测试模型
    test_loss, test_ACC = test(epoch)

    train_logger.info(f"{epoch} {train_loss} {train_ACC} {test_loss} {test_ACC}\n")
    writer_train.add_scalar("Loss", train_loss, epoch)
    writer_test.add_scalar("Loss", test_loss, epoch)
    writer_train.add_scalar("Acc", train_ACC, epoch)
    writer_test.add_scalar("Acc", test_ACC, epoch)


best_model_ACC = 0


def test(epoch):
    model.eval()
    test_loss = 0.
    test_ACC = 0.
    true_label = []
    pre_label = []

    test_result = []
    for batch_idx, (data, bag_label) in enumerate(test_loader):
        true_label.append(bag_label.cpu().detach().numpy()[0])
        # instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        data, bag_label=data.squeeze(0), bag_label.squeeze(0)
        Y_prob = model(data)
        test_result.append(Y_prob.cpu().detach().numpy())  # 用于记录测试的输出结果 最后一个是label
        loss = train_loss_fn(Y_prob, bag_label)
        _, Y_prob = torch.max(Y_prob, 1)
        ACC = Y_prob.eq(bag_label).cpu().float().mean().item()

        test_loss += loss.item()
        pre_label.append(Y_prob.cpu().numpy()[0])
        test_ACC += ACC

        # if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
        #     bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
        #     instance_level = list(zip(instance_labels.numpy()[0].tolist(),
        #                          np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

        # logging.info('\nTrue Bag Label, Predicted Bag Label: {}\n'
        #       'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))
    # 记录测试结果
    test_result = np.array(test_result)
    test_true_label = np.array(true_label)

    np.savez(
        os.path.join(
            project_path,
            f"test_results/{epoch}_test_result.npz"),
        test_result=test_result,
        test_true_label=test_true_label)

    test_ACC /= len(test_loader)
    test_loss /= len(test_loader)

    class_report = classification_report(true_label, pre_label, digits=4)  # 分类报告
    cm = confusion_matrix(true_label, pre_label)
    test_logger.info(f"******************epoch:{epoch}****************")
    test_logger.info(class_report)
    test_logger.info(cm)
    plot_confusion_matrix(cm, os.path.join(project_path, f"results/{epoch}.png"), classes=LABELS)

    global best_model_ACC
    if (test_ACC > best_model_ACC):
        best_model_ACC = test_ACC
        torch.save(model.state_dict(), os.path.join(project_path, f"best.pth"))
    torch.save(model.state_dict(), os.path.join(project_path, f"last.pth"))

    return test_loss, test_ACC


if __name__ == "__main__":
    logging.info('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
