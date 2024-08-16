import torch
from torch.nn.functional import mse_loss, relu

def variance(z, gamma=1):
    return relu(gamma - z.std(0)).mean()

def invariance(z1, z2):
    return mse_loss(z1, z2)

# 데이터의 공분산 행렬을 사용하여 off-diagonal 공분산 제곱의 평균을 측정. 데이터의 공분산 구조에서 대각선 요소를 제외한 off-diagonal 요소의 기여도를 평가하는데 도움이 될 수 있음
def covariance(z):
    n, d = z.shape # n : 데이터 포인트의 수(행의 수), d : 특성의 수 (열의 수)
    mu = z.mean(0) # 각 열(특성)에 대해 평균을 계산하여 mu라는 1차원 텐서로 저장. mu는 각 특성의 평균을 담고 있음
    
    # z - mu를 이용하여 각 데이터 포인트에서 평균을 뺀 중심화된 데이터 포인트를 생성
    # 두 중심화된 데이터 포인트 행렬의 외적을 계산하여 공분산 행렬을 구함
    # "ni,nj->ij"는 z-mu의 각 행과 열의 곱을 합산하여 결과를 ij 형태의 공분산 행렬로 계산함
    # 마지막으로 이 공분산 행렬을 (n-1)로 나누어서 샘플 공분산을 계산
    cov = torch.einsum("ni,nj->ij", z-mu, z-mu) / (n - 1)
    off_diag = cov.pow(2).sum() - cov.pow(2).diag().sum() # off_diag는 대각선 요소를 제외한 off-diagonal요소(대각선이 아닌 요소)의 제곱합을 계산한 것 (전체 요소의 제곱합에서 대각선 요소의 제곱합을 빼줬음)
    return off_diag / d # off-diagonal 공분산 제곱의 합을 d로 나누어서 평균을 구함. d는 특성의 수 이므로 각 특성에 대한 off-diagonal 공분산 제곱의 평균을 구한 것 임
