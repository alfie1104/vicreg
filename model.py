import torch
from torch import nn
from torchvision.models.resnet import resnet18

class Projector(nn.Module):
    def __init__(self, encoder_dim, projector_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(encoder_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim)
        )
            
    def forward(self, x):
        return self.network(x)
    
class VICReg(nn.Module):
    def __init__(self, encoder_dim, projector_dim):
        super().__init__()
        
        # the default ResNet has a 7x7 kernel with stride 2 as its initial
        # convolutional layer. this works for ImageNet but is too reductive for
        # CIFAR-10. we follow the SimCLR paper and replace it with a 3x3 kernel
        # with stride 1 and remove the max pooling layer.
        
        self.encoder = resnet18(num_classes=encoder_dim)
        # Default 커널 7x7대비 커널 크기를 줄여서 더 세밀한 지역적 특징을 학습할 수 있게 구성
        # Default stride = 2 대비 stride=1로 적용하여 입력 이미지의 해상도를 유지(이미지의 공간적 정보를 더 많이 유지)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=1)
        # 원래 Resnet18의 경우 conv1 다음 max pooling layer가 있지만 제거하였음(Identity로 변경) -> 해상도가 유지됨으로써 보다 세밀한 정보를 다루게 됨
        self.encoder.maxpool = nn.Identity()
        
        self.projector = Projector(encoder_dim, projector_dim)
    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2)) # x1과 x2를 배치 차원에서 합쳐서 x1의 배치 수 + x2의 배치 수를 갖는 x를 구성
        y = self.encoder(x) # 합쳐진텐서 x를 encoder로 처리하여 y텐서 획득

        # PyTorch.chunk(n)은 텐서를 주어진 개수 n개의 조각으로 나누는 메서드임
        # self.projector(y)는 y텐서에 대해 프로젝터를 적용함으로써 y의 특성(feature)를 변환한 새로운 텐서가 생성됨
        # 이렇게 프로젝션된 텐서를 두개의 청크로 나누는 역할을 함
        # 즉, 합쳐진 x1과 x2에 해당하는 특성들을 다시 분리하기 위해서 사용
        # (chunk의 2번째 인자로 dim을 넣을 수 있는데 안 넣었으므로 기본값인 0이 입력되고 배치차원으로 자르게 됨)
        return self.projector(y).chunk(2)
