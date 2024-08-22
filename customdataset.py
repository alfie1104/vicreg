import torchvision.datasets as datasets # 이미지 데이터셋 집합체
import torchvision.transforms as transforms # 이미지 변환 툴

from torch.utils.data import DataLoader, Dataset # 학습 및 배치로 모델에 넣어주기 위한 툴
from PIL import Image
import cv2

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, train_mode = True, transforms = None):
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list

    def __getitem__(self, index): # index번째 data를 return
        img_path = self.img_path_list[index]

        # Get Image data
        image = cv2.imread(img_path) # [height, width, channel] 형태로 데이터를 읽게됨. 여기서 channel은 rgb색상을 표현하는 BGR 채널임(RGB가 아님) (두번째 파라미터로 -1을 넘기면 alpha값까지 읽게 됨)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB 색상으로 변환
        #image = Image.open(img_path) # [height, width, channel] 형태로 데이터를 읽게 됨 (그런데 PIL은 이미지가 RGBA일 경우 기본적으로 RGBA로 불러와서 색상채널이 4채널이 됨)

        # PIL로 불러와도 되고 cv2로 불러와도 되는데 cv2로 불러오면 numpy array로 불러오고, PIL로 불러오면 PIL 객체로 불러옴
        # augmentation할때 transforms에서 PIL로 불러온 이미지는 PILToTensor를
        # cv2로 불러온 이미지는 ToTensor를 적용해서 변환해줘야함
        # 일반적으로 PIL로 불러오는게 빠른데 png는 cv2가 빠름
        
        # Tensor로 바꾸면 [channel, height, width] 형태가 됨

        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)