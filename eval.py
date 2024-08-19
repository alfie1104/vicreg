import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import transforms
from model import VICReg
from tqdm import tqdm

NUM_EPOCHS = 50 # downstream task용 layer학습을 위한 epoch 수
BATCH_SIZE = 256
LEARNING_RATE = 0.02
WEIGHT_DECAY = 1e-6
CHECK_POINT = "checkpoint_500epoch.pt"
NUM_CLASSES = 10

# load model checkpoint and move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cp = torch.load(CHECK_POINT, map_location=device, weights_only=False)
encoder_dim, projector_dim = cp["encoder_dim"], cp["projector_dim"]
model = VICReg(encoder_dim, projector_dim).to(device)
model.load_state_dict(cp["model_state_dict"])
model.eval()

# create linear layer, optimizer, scheduler and training hyperparams
num_classes, batch_size, num_epochs = NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS
linear = nn.Linear(encoder_dim, num_classes).to(device) # downstream task용 layer
opt = SGD(linear.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(opt, num_epochs)

# data augmentations used to regularize the linear layer
augment = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# define train and test datasets
train_data = CIFAR10(root=".", train=True, download=True, transform=augment)
test_data = CIFAR10(root=".", train=False, download=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size)

# use standard cross entropy loss
criterion = nn.CrossEntropyLoss()
progress = tqdm(range(num_epochs))

# train the linear layer on the representations of the frozen encoder
for epoch in progress:
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        encoder_out = model.encoder(images)
        preds = linear(encoder_out)
        loss = criterion(preds, labels)
        loss.backward()
        # 엔코더 부분인 VICReg model은 eval로 설정하여 가중치 업데이트를 하지 않으며, 
        # optimizer는 생성 시 downstream task용 linear 모델의 parameters에 대해서 생성하였으므로 linear 모델에 대해서만 가중치 업데이트를 진행하게됨
        opt.step()
        opt.zero_grad()
        progress.set_description(f"Loss: {loss.item()}, epoch : {epoch}")
    scheduler.step()

# evaluate the accuracy after training the linear layer
num_correct = len(test_data)
for image, label in test_dataloader:
    image, label = image.to(device), label.to(device)
    encoder_out = model.encoder(image)
    pred = linear(encoder_out).argmax(-1)
    num_incorrect = torch.count_nonzero(pred-label)
    num_correct -= num_incorrect
print("Accuracy:", num_correct / len(test_data))
