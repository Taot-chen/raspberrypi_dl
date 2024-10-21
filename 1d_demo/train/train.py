import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset,random_split
from gen_dataset import package_dataset
from Models.MobileNetV3 import MobileNetV3_large

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

def train(model, epoch, dataloader, testloader, device, criterion, optimizer, train_acc_list,     test_acc_list, show_result_epoch):
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
        test(model, testloader, device, criterion, optimizer, test_acc_list)

def main():
    data_path = "./dataset/"
    data = np.load(f'{data_path}/data.npy')
    label = np.load(f'{data_path}/label.npy')
    data_part = 0.7
    epoch_num = 1000
    show_result_epoch = 10 
    bsz = 200
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
    model =MobileNetV3_large(in_channels=data_chal, classes=classes)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_acc_list = []
    test_acc_list = []

    for epoch in range(epoch_num):
        train(model, epoch, dataloader, testloader, device, criterion, optimizer, train_acc_list, test_acc_list, show_result_epoch)

    plt.plot(np.array(range(epoch_num//show_result_epoch)) * show_result_epoch, train_acc_list)
    plt.plot(np.array(range(epoch_num//show_result_epoch)) * show_result_epoch, test_acc_list)
    plt.legend(['train', 'test'])
    plt.title('Result')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig("./result.png")

    
if __name__ == "__main__":
    main()
