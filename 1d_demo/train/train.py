import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset,random_split
from gen_dataset import package_dataset
from Models.MobileNetV3 import MobileNetV3_large, MobileNetV3_small
from Models.AlexNet import AlexNet
from Models.DenseNet import DenseNet
from Models.GoogleLeNet import GoogLeNet
from Models.LeNet import LeNet
from Models.Mnasnet import MnasNetA1
from Models.MobileNetV1 import MobileNetV1
from Models.MobileNetV2 import MobileNetV2
from Models.ResNet import ResNet50
from Models.shuffuleNetV1 import shuffuleNetV1_G3
from Models.shuffuleNetV2 import shuffuleNetV2
from Models.SqueezeNet import SqueezeNet
from Models.VGG import VGG19
import argparse

class Dataset(Dataset):
    def __init__(self, data):
        self.len = len(data)
        self.x_data = torch.from_numpy(np.array(list(map(lambda x: x[0], data)), dtype=np.float32))
        self.y_data = torch.from_numpy(np.array(list(map(lambda x: x[-1], data)))).squeeze().long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def test(model, testloader, device, criterion, optimizer, test_acc_list):
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for testdata in testloader:
            test_data_value, test_data_label = testdata
            test_data_value, test_data_label = test_data_value.to(device), test_data_label.to(device)
            test_data_label_pred = model(test_data_value)
            test_probability, test_predicted = torch.max(test_data_label_pred.data, dim=1)
            test_total += test_data_label_pred.size(0)
            test_correct += (test_predicted == test_data_label).sum().item()
    test_acc = round(100 * test_correct / test_total, 3)
    test_acc_list.append(test_acc)
    print(f'Test accuracy:{(test_acc)}%')
    return test_acc

def train(model, epoch, dataloader, testloader, device, criterion, optimizer, train_acc_list, test_acc_list, show_result_epoch, best_avg_score, model_name):
    model.train()
    train_correct = 0
    train_total = 0
    for data in dataloader:
        train_data_value, train_data_label = data
        train_data_value, train_data_label = train_data_value.to(device), train_data_label.to(device)
        train_data_label_pred = model(train_data_value)
        loss = criterion(train_data_label_pred, train_data_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % show_result_epoch == 0:
        probability, predicted = torch.max(train_data_label_pred.data, dim=1)
        train_total += train_data_label_pred.size(0)
        train_correct += (predicted == train_data_label).sum().item()
        train_acc = round(100 * train_correct / train_total, 4)
        train_acc_list.append(train_acc)
        print('=' * 10, epoch // 10, '=' * 10)
        print('loss:', loss.item())
        print(f'Train accuracy:{train_acc}%')

        test_acc = test(model, testloader, device, criterion, optimizer, test_acc_list)
        if test_acc > best_avg_score:
            best_avg_score = test_acc
            print('saving new weights...\n')
            torch.save(model.state_dict(), f'{model_name}_epoch_{epoch + 1}_valid_avg_{test_acc:0.4f}_model_weights.pth')
    return best_avg_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="MobileNetV3_large", type=str)
    parser.add_argument("--all", default=0, type=int)
    parser.add_argument("--model_supported", default=0, type=int)
    args = parser.parse_args()

    supported_model_dict = {
        "MobileNetV3_large": MobileNetV3_large,
        "MobileNetV3_small": MobileNetV3_small,
        "AlexNet": AlexNet,
        "DenseNet": DenseNet,
        "GoogleLeNet": GoogLeNet,
        "LeNet": LeNet,
        "MnasNetA1": MnasNetA1,
        "MobileNetV1": MobileNetV1,
        "MobileNetV2": MobileNetV2,
        "ResNet50": ResNet50,
        "shuffuleNetV1_G3": shuffuleNetV1_G3,
        "shuffuleNetV2": shuffuleNetV2,
        "SqueezeNet": SqueezeNet,
        "VGG19": VGG19
    }
    if args.model_supported:
        print("Supported models:\n")
        for item in supported_model_dict.keys():
            print(item)
        return 0
    model_name = args.model_name
    assert model_name in supported_model_dict.keys(), f"{model_name} hasn't been supported yet"

    data_path = "./dataset/"
    data = np.load(f'{data_path}/data.npy')
    label = np.load(f'{data_path}/label.npy')
    data_part = 0.7
    epoch_num = 100
    show_result_epoch = 10 
    bsz = 50
    dataset, data_chal, data_len, classes = package_dataset(data, label)
    # partition dataset
    train_len = int(len(dataset) * data_part)
    test_len = int(len(dataset)) - train_len
    train_data, test_data = random_split(dataset=dataset, lengths=[train_len, test_len])
    train_dataset = Dataset(train_data)
    test_dataset = Dataset(test_data)
    dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bsz)
    testloader = DataLoader(test_dataset, shuffle=True, batch_size=bsz)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mdoel = None
    if not args.all and model_name:
        model = supported_model_dict[model_name]()
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        criterion.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        train_acc_list = []
        test_acc_list = []
        best_avg_score = 0
        for epoch in range(epoch_num):
            best_avg_score = train(model, epoch, dataloader, testloader, device, criterion, optimizer, train_acc_list, test_acc_list, show_result_epoch, best_avg_score, model_name)
        plt.plot(np.array(range(epoch_num//show_result_epoch)) * show_result_epoch, train_acc_list)
        plt.plot(np.array(range(epoch_num//show_result_epoch)) * show_result_epoch, test_acc_list)
        plt.legend(['train', 'test'])
        plt.title('Result')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(f"./{model_name}_result.png")
    elif args.all:
        for model_name in supported_model_dict.keys():
            model = supported_model_dict[model_name]()
            model.to(device)
            print("model: ", model)
            # criterion = torch.nn.CrossEntropyLoss()
            # criterion.to(device)
            # # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            # train_acc_list = []
            # test_acc_list = []
            # best_avg_score = 0
            # for epoch in range(epoch_num):
            #     best_avg_score = train(model, epoch, dataloader, testloader, device, criterion, optimizer, train_acc_list, test_acc_list, show_result_epoch, best_avg_score, model_name)
            # plt.plot(np.array(range(epoch_num//show_result_epoch)) * show_result_epoch, train_acc_list)
            # plt.plot(np.array(range(epoch_num//show_result_epoch)) * show_result_epoch, test_acc_list)
            # plt.legend(['train', 'test'])
            # plt.title('Result')
            # plt.xlabel('Epoch')
            # plt.ylabel('Accuracy')
            # plt.savefig(f"./{model_name}_result.png")

if __name__ == "__main__":
    main()
