import os
import torch
import torch.nn as nn
import torchvision
import utils
import time
import matplotlib.pyplot as plt
import numpy as np
 
 
def load_model(class_num=27):
    mobilenet = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
    mobilenet.classifier[1] = nn.Linear(in_features=mobilenet.classifier[1].in_features, out_features=class_num)
    return mobilenet
 
def train(epochs):
    train_loader, val_loader, class_names = utils.data_load("./synthetic-asl-alphabet_dataset/Train_Alphabet", "./synthetic-asl-alphabet_dataset/Test_Alphabet", 224, 224, 256)
    print("类别名称：", class_names)
    model = load_model(class_num=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if os.path.exists("mobilenet_latest.pth"):
        if torch.cuda.is_available():
            model.load_state_dict(torch.load("mobilenet_latest.pth"))
        else:
            model.load_state_dict(torch.load("mobilenet_latest.pth", map_location=torch.device('cpu')))
        print("加载已有模型继续训练")
 
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    patience = 5
    patience_counter = 0
    epoch_list = []

    start = time.time()
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        epoch_list.append(epoch + 1)
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct / total)
        print("训练损失：{:.4f}，准确率：{:.4f}".format(train_losses[-1], train_accuracies[-1]))
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_losses.append(running_loss / len(val_loader))
        val_accuracies.append(correct / total)
        print("验证损失：{:.4f}，准确率：{:.4f}".format(val_losses[-1], val_accuracies[-1]))
        if val_accuracies[-1] > best_val_accuracy:
            best_val_accuracy = val_accuracies[-1]
            torch.save(model.state_dict(), "mobilenet_latest.pth")
            print("模型已保存")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发，停止训练")
                break
    end = time.time()
    print("train total time: {}s".format(end - start))
    plt.plot(epoch_list, train_accuracies)
    plt.plot(epoch_list, val_accuracies)
    plt.legend(['train', 'test'])
    plt.title('Result')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f"./mobilenet_latest.png")

if __name__ == '__main__':
    train(epochs=50)
