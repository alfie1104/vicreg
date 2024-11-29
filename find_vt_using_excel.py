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
from utils import find_similar_rows

import re


# 파일 이름에서 숫자를 추출하는 함수
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

# embeding 결과가 저장된 excel 파일 불러오기
file_dir = "./embeded_result"

# file_dir 내 엑셀 파일 목록을 가져와서 숫자순으로 정렬
excel_fies = [f for f in os.listdir(file_dir) if f.endswith(".xlsx") and f.startswith("embeded")]
sorted_files = sorted(excel_fies, key=extract_number)

# 빈 리스트를 생성하여 각 파일의 데이터프레임을 저장
dfs = []

# 정렬된 파일 목록을 순회하며 데잍를 읽어들임
for filename in sorted_files:
    print(f"{filename} is being appended")
    file_path = os.path.join(file_dir, filename)
    df = pd.read_excel(file_path)
    dfs.append(df)

# 모든 데이터 프레임을 하나로 통합
df_embeded_results = pd.concat(dfs, ignore_index=True)

# 유사한 이미지 찾기 테스트
target_index = 50
top_k = 10
is_using_db = False


if is_using_db:
    df_embeded_results = df_embeded_results.drop('image_index', axis=1)
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
    print(f"DB_LCTN : {df_embeded_results.iloc[target_index]["DB_LCTN"]}, ID_VT : {df_embeded_results.iloc[target_index]["ID_VT"]}, HDDN_RVSN : {df_embeded_results.iloc[target_index]["HDDN_RVSN"]}, {df_embeded_results["image_index"][target_index]}")
    results = find_similar_rows(df_embeded_results, target_index, top_k)
    for idx, similarity in results:
        current = df_embeded_results.iloc[idx]
        print(f"{current["DB_LCTN"]}-{current["ID_VT"]}-{current["HDDN_RVSN"]}, {current["image_index"]}, {similarity}")    
