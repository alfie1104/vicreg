import os
import torch

# 코사인 유사도 계산을 위한 라이브러리
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def save_model(model, optimizer, epoch, name, encoder_dim, projector_dim, scheduler = None):
    out = os.path.join("./saved_models/", name.format(epoch))

    torch.save({
        "epoch": epoch + 1,
        "encoder_dim": encoder_dim,
        "projector_dim": projector_dim,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler != None else {}
    },out)

def find_similar_rows(df : pd.DataFrame, target_row_index : int, top_k = 1):
    """
    cosine 유사도를 기반으로 df에서 df[target_row_index] 데이터와 유사한 행을 찾아서 반환

    Parameters
    -----------
    df : pd.DataFrame
        유사도 탐색 대상 데이터들
    target_row_index : int
        유사도 비교 기준 데이터가 위치한 행의 인덱스
    top_k : int, optional
        몇 개의 유사 데이터를 반환할지 선택. 기본값은 1

    Returns:
    --------
    list[tuple[Any, Any]]
        top_k개의 [데이터 인덱스 번호, 유사도]를 갖는 list를 반환
    """
     # 특정 열만 선택 (Image Name 열 제외)
    feature_columns = [col for col in df.columns if not col.startswith('Image')]
    features = df[feature_columns].values

    # 타겟 행의 특징 벡터
    target_vector = features[target_row_index].reshape(1, -1)
    
    # 모든 행과의 코사인 유사도 계산
    similarities = cosine_similarity(target_vector, features)

    # 유사도가 가장 높은 top_k개의 인덱스 찾기
    similarities = similarities[0]
    similarities[target_row_index] = -1 # 타겟 행 제외

    most_similar_indices = np.argsort(similarities)[::-1][:top_k]

    # 결과를 (인덱스, 유사도) 튜플의 리스트로 반환
    results = [(idx, similarities[idx]) for idx in most_similar_indices]

    return results