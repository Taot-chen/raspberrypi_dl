import numpy as np
import pathlib

def gen_data(data_num = 100, data_chal = 3, data_len = 224, classes = 2, save_path="./dataset/"):
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 
    data = np.random.randn(data_num, data_chal, data_len)
    label = np.random.randint(0, classes, data_num)
    np.save(f'{save_path}/data.npy', data, allow_pickle=True)
    np.save(f'{save_path}/label.npy', label, allow_pickle=True)

def package_dataset(data, label):
    dataset = [[i, j] for i, j in zip(data, label)]
    data_chal = data[0].shape[0]
    data_len = data[0].shape[1]
    classes = len(np.unique(label))
    return dataset, data_chal, data_len, classes


if __name__ == '__main__':
    data_path = "./dataset/"
    data_num = 100
    data_chal = 3
    data_len = 224
    classes = 2
    gen_data(data_num = data_num, data_chal = data_chal, data_len = data_len, classes = classes, save_path = data_path)
    data = np.load(f'{data_path}/data.npy')
    label = np.load(f'{data_path}/label.npy')
    dataset, channels, length, classes = package_dataset(data, label)
    print("generate dataset complete")
    print(channels, length, classes)
