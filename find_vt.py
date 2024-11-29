import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import VICReg
import os

# custom dataset 및 목록 관련 라이브러리
from customdataset import CustomDataset
import pandas as pd
from glob import glob

from vector_db import search_similar_vt_image


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
        #transforms.RandomResizedCrop((32,32), scale=(0.3, 1.0)), # resnet18 모델에 맞는 크기로 이미지 크기 조정 필요 # scale에 지정한 범위내의 크기로 자름
        transforms.RandomResizedCrop((32,32), scale=(1.0, 1.0)), # 학습과 다르게 테스트 시점에는 이미즈 크기만 축소함
    ])

    # 이미지 데이터 획득
    data_dir = "D:\\project_javascript\\vt-image-downloader"
    img_list = pd.read_excel(os.path.join(data_dir, "vt-image-list.xlsx"))

    # 이미지 폴더에서 데이터 로드
    all_img_path = []
    all_img_path.extend(glob(os.path.join(data_dir,"images","*.png")))
    all_img_path.sort(key=lambda x: int(os.path.basename(x).split(".")[0])) # 파일명이 1213.png 형태이므로 .앞부분의 숫자만 가져와서 숫자형태로 바꾼 뒤 정렬

    # 데이터 loader 구성(BATCH_SIZE 만큼 한번에 읽어오도록)
    data_len = len(all_img_path)

    dataset = CustomDataset(all_img_path, [], train_mode=False, transforms=augment)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=False)

    # 이미지 데이터들을 model에 넣어서 벡터로 변환
    # 변환 결과(projector_dim의 차원)를 Vector DB에 기록
    embeded_results = [] # 변환 결과를 저장할 리스트

    with torch.no_grad():
        for i, (image, _) in enumerate(dataloader):

            image = image.to(device)
            encoder_out = model.encoder(image)
            # print(encoder_out.shape) # torch.Size([256, 512])
            projector_out = model.projector(encoder_out)
            # print(projector_out.shape) # torch.Size([256, 1024])

            # tensor형태의 projector_out을 numpy 배열로 변환
            np_projector_out = projector_out.cpu().numpy()

            # 결과를 리스트에 추가
            for j, output in enumerate(np_projector_out):
                img_path = all_img_path[i * BATCH_SIZE + j]
                image_index = os.path.basename(img_path).split(".")[0]
                embeded_results.append([image_index] + output.tolist())
            
            print(f"{i+1}th batch datas are being embeded...")
            if i == 23:
                # 테스트로 256 * 23개만 저장
                break
    
    # 결과를 Data Frame으로 변환
    columns = ['image_index'] + [f"feature_{i}" for i in range(projector_dim)]
    df_embeded_results = pd.DataFrame(embeded_results, columns=columns)

    # img_list(index, dbLctn, idVt, hddnRvsn, error 가 기록된 데이터 프레임)와 df_embeded_results를 병합
    img_list = img_list[['index','dbLctn','idVt','hddnRvsn']].rename(columns={"dbLctn":"DB_LCTN", "idVt":"ID_VT","hddnRvsn":"HDDN_RVSN"}) #img_list = img_list.drop('error', axis=1) 이렇게 해도 되지만 직관성을 위해 직접 칼럼명을 지정하였음

    df_embeded_results["image_index"] = df_embeded_results["image_index"].astype(int)
    
    df_embeded_results = pd.merge(img_list, df_embeded_results, left_on='index', right_on='image_index', how='inner')
    df_embeded_results = df_embeded_results.drop('image_index', axis=1)
    df_embeded_results = df_embeded_results.drop('index', axis=1)

    # 유사한 이미지 찾기 테스트
    # target_index = 50
    target_index = 5758
    top_k = 10

    results = search_similar_vt_image(
        query_embedding=df_embeded_results.iloc[target_index, 3:].values.tolist(),
        tok_k=10, 
        db_lctn=df_embeded_results.iloc[target_index]["DB_LCTN"],
        id_vt=df_embeded_results.iloc[target_index]["ID_VT"],
        hddn_rvsn=df_embeded_results.iloc[target_index]["HDDN_RVSN"]
    )
    print(f"DB_LCTN : {df_embeded_results.iloc[target_index]["DB_LCTN"]}, ID_VT : {df_embeded_results.iloc[target_index]["ID_VT"]}, HDDN_RVSN : {df_embeded_results.iloc[target_index]["HDDN_RVSN"]}")
    print(results)    
else:
    print("Model checkpoint doesn't exist")
