import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import VICReg
import os

# custom dataset 및 목록 관련 라이브러리
from customdataset import CustomDataset
import pandas as pd
from glob import glob

BATCH_SIZE = 256
LEARNING_RATE = 0.02
WEIGHT_DECAY = 1e-6
#CHECK_POINT = "checkpoint_500epoch.pt"
CHECK_POINT = os.path.join("./saved_models/", "VICReg_Custom_RN18_P128_LR2e4_WD1e6_B256_checkpoint_400_20240816.pt")

# model checkpoint 불러오기 및 model을 GPU에 할당
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists(CHECK_POINT):
    print("Loading from checkpoint...")

    # checkpoint 불러오기
    cp = torch.load(CHECK_POINT, weights_only=False, map_location=device)

    # model 정의
    encoder_dim, projector_dim = cp["encoder_dim"], cp["projector_dim"]
    model = VICReg(encoder_dim, projector_dim).to(device)

    # checkpoint에서 model 상태 획득 및 evaluation 모드로 설정
    model.load_state_dict(cp["model_state_dict"])
    model.eval()

    # data augmentations used to regularize the linear layer
    augment = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop((32,32), scale=(0.3, 1.0)), # resnet18 모델에 맞는 크기로 이미지 크기 조정 필요 # scale에 지정한 범위내의 크기로 자름
    ])

    # 이미지 데이터 획득
    data_dir = "D:\\project_javascript\\vt-image-downloader"
    img_list = pd.read_excel(os.path.join(data_dir, "vt-image-list.xlsx"))

    # 이미지 폴더에서 데이터 로드
    all_img_path = []
    all_img_path.extend(glob(os.path.join(data_dir,"images","*.png")))
    all_img_path.sort(key=lambda x: int(os.path.basename(x).split(".")[0])) # 파일명이 1213.png 형태이므로 .앞부분의 숫자만 가져와서 숫자형태로 바꾼 뒤 정렬

    # Test용 데이터 설정
    train_len = int(len(all_img_path)*0.8)
    test_len = int(len(all_img_path)*0.2)

    test_img_path = all_img_path[train_len:]
    test_dataset = CustomDataset(test_img_path, [], train_mode=False, transforms=augment)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    # Test용 이미지 데이터들을 model에 넣어서 벡터로 변환
    # 변환 결과(projector_dim의 차원)를 excel에 기록
    embeded_results = [] # 변환 결과를 저장할 리스트

    with torch.no_grad():
        for i, (image, _) in enumerate(test_dataloader):

            image = image.to(device)
            encoder_out = model.encoder(image)
            # print(encoder_out.shape) # torch.Size([256, 512])
            projector_out = model.projector(encoder_out)
            # print(projector_out.shape) # torch.Size([256, 1024])

            # tensor형태의 projector_out을 numpy 배열로 변환
            np_projector_out = projector_out.cpu().numpy()

            # 결과를 리스트에 추가
            for j, output in enumerate(np_projector_out):
                img_path = test_img_path[i * BATCH_SIZE + j]
                img_name = os.path.basename(img_path)
                embeded_results.append([img_name] + output.tolist())

            if i == 1:
                # 테스트로 256 * 2개만 저장
                break
    
    # 결과를 Data Frame으로 변환
    columns = ['Image Name'] + [f"Feature_{i}" for i in range(projector_dim)]
    df_embeded_results = pd.DataFrame(embeded_results, columns=columns)

    # Excel 파일로 저장
    output_dir = "./"
    output_file = os.path.join(output_dir, "embeded_result.xlsx")
    df_embeded_results.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")
else:
    print("Model checkpoint doesn't exist")
