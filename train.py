import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import transforms
from tqdm import tqdm
from model import VICReg
from loss import variance, invariance, covariance
from utils import save_model


class Augmentation:
    """
    Wrapper around a PyTorch transform, outputting two different augmentations
    for a single input. Applying this when loading a dataset ensures that a
    dataloader will provide two augmentations for each sample in a batch.
    """
    augment = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.RandomSolarize(0.5, p=0.2),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
        
    def __call__(self, x):
        return self.augment(x), self.augment(x)

def main():
    NUM_EPOCHS = 500
    BATCH_SIZE = 256
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-6
    NUM_WORKERS = 2

    # define model and move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_dim, projector_dim = 512, 1024
    model = VICReg(encoder_dim, projector_dim).to(device)

    # prepare data, optimizer, and training hyperparams
    num_epochs, batch_size = NUM_EPOCHS, BATCH_SIZE
    data = CIFAR10(root=".", train=True, download=True, transform=Augmentation())
    dataloader = DataLoader(data, batch_size, shuffle=True, num_workers=NUM_WORKERS)
    opt = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    progress = tqdm(range(num_epochs))

    # load from checkpoint if it exists
    if os.path.exists("checkpoint.pt"):
        print("Loading from checkpoint...")
        cp = torch.load("checkpoint.pt", weights_only=False)
        model.load_state_dict(cp["model_state_dict"])
        opt.load_state_dict(cp["optimizer_state_dict"])
        progress = tqdm(range(cp["epoch"], num_epochs))

    # train the model and regularly save to disk
    for epoch in progress:
        for images, _ in dataloader:
            x1, x2 = [x.to(device) for x in images]
            z1, z2 = model(x1, x2)
            
            la, mu, nu = 25, 25, 1 # 논문에서 계수들을 이 값으로 설정하였음.(nu를 1로 설정한 뒤 lambda와 mu에 대해 grid search를 진행. 이때 조건으로 lambda = nu > 1이 되도록 하였음)

            var1, var2 = variance(z1), variance(z2)
            inv = invariance(z1, z2)
            cov1, cov2 = covariance(z1), covariance(z2)

            # variance, invariance, covariance를 이용하여 loss 구성
            # variance : 각 임베딩 차원의 분산을 0보다 크게 해서 collapse 방지
            # invariance : 하나의 객체에 서로 다른 augmentation을 적용해도 의미론적으로 같은 값이 되도록 학습시키기 위해 출력된 임베딩 벡터 쌍에 MSE Loss 적용
            # covariance : 하나의 임베딩 결과에서 임베딩 벡터의 차원간의 상관성을 최소화하여 독립적인 정보가 인코딩 되도록 구성
            loss = la*inv + mu*(var1 + var2) + nu*(cov1 + cov2)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            progress.set_description(f"Loss: {loss.item()}, epoch : {epoch}")

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            # 파일명 예 : VICReg_CIFAR10_RN18_P128_LR2e4_WD1e6_B256_checkpoint_{}_20240816.pt
            # VICReg : 모델
            # CIFAR10 : 데이터셋
            # RN18 : ResNet18 신경망 아키텍처(18개의 레이어를 가진 Residual Network)
            # P128 : 모델의 패치 크기 또는 이미지 크기
            # LR2e4 : 학습률이 0.0002
            # WD1e6 : weight decay 0.000006
            # B256 : batch_size = 256
            # checkpoint_{} : 체크포인트 파일 이름의 형식. 중괄호 {}는 모델의 훈련 중 저장된 다양한 체크포인트를 식별하는 자리 표시자
            # 20240816 : 파일이 저장된 날짜
            # .pt : PyTorch에서 저장된 모델 파일의 확장자
            save_model(
                model=model,
                optimizer=opt,
                epoch=epoch,
                name="VICReg_CIFAR10_RN18_P128_LR2e4_WD1e6_B256_checkpoint_{}_20240816.pt",
                encoder_dim=encoder_dim,
                projector_dim=projector_dim,

            )

if __name__ == "__main__":
    main()