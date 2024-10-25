import json
from torchvision.models import mobilenet_v2
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from PIL import Image
import time
 
class SignLanguageRecognition:
    def __init__(self, module_file='./mobilenet_latest.pth', labels_file='./labels.json'):
        self.module_file = module_file
        self.CUDA = torch.cuda.is_available()
        self.net = mobilenet_v2(num_classes=27)
        if self.CUDA:
            self.net.cuda()
        self.net.load_state_dict(torch.load(self.module_file, map_location='cuda' if self.CUDA else 'cpu'))
        self.net.eval()
        self.labels = self.load_labels(labels_file)

    def load_labels(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
 
    @torch.no_grad()
    def preprocess_image(self, image_stream):
        img = Image.open(image_stream)
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.56719673, 0.5293289, 0.48351972], std=[0.20874391, 0.21455203, 0.22451781]),
        ])
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        if self.CUDA:
            img = img.cuda()
        return img

    @torch.no_grad()
    def recognize(self, image_stream):
        img = self.preprocess_image(image_stream)
        y = self.net(img)
        y = F.softmax(y, dim=1)
        p, cls_idx = torch.max(y, dim=1)
        return y.cpu(), cls_idx.cpu()
 
    def predict(self, image_stream):
        probs, cls = self.recognize(image_stream)
        _, cls = torch.max(probs, 1)
        p = probs[0][cls.item()]
        cls_index = str(cls.numpy()[0])
        label_name = self.labels.get(cls_index, "未知标签")
        return label_name, p.item()

if __name__ == "__main__": 
    recongize=SignLanguageRecognition()
    start = time.time()
    ret = recongize.predict('./synthetic-asl-alphabet_dataset/Test_Alphabet/A/e4761e88-10df-41d2-b980-463375c5c46c.rgb_0000.png')
    print(ret)
    end = time.time()
    print("infer total time: {}s".format(end - start))

    start = time.time()
    ret = recongize.predict('./synthetic-asl-alphabet_dataset/Test_Alphabet/Blank/5da53b24-d860-4921-a645-9fe85bd91213.rgb_0000.png')
    print(ret)
    end = time.time()
    print("infer total time: {}s".format(end - start))
