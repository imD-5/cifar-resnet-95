import random
import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from tqdm import tqdm

def unpickle(file):
    """Load CIFAR-10 batch file."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir (str): Directory with all the CIFAR-10 batch files
            train (bool): If True, creates dataset from training files, otherwise from test file
            transform (callable, optional): Optional transform to be applied to samples
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []

        if self.train:
            # Load training batches
            for i in range(1, 6):
                file_path = os.path.join(root_dir, f'data_batch_{i}')
                entry = unpickle(file_path)
                self.data.append(entry[b'data'])
                self.targets.extend(entry[b'labels'])

            self.data = np.vstack(self.data)
        else:
            # Load test batch
            file_path = os.path.join(root_dir, 'test_batch')
            entry = unpickle(file_path)
            self.data = entry[b'data']
            self.targets = entry[b'labels']

        # Load label names
        meta_file = os.path.join(root_dir, 'batches.meta')
        meta_data = unpickle(meta_file)
        self.classes = [label.decode('utf-8') for label in meta_data[b'label_names']]

        # Reshape data to 32x32x3 images
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        # Convert to PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

def create_dataloaders(root_dir, batch_size=64, val_size=500, transform_train=None, transform_test=None):
    """
    Create train, validation, and test dataloaders for CIFAR-10.
    
    Args:
        root_dir (str): Directory containing CIFAR-10 data files
        batch_size (int): Batch size for dataloaders
        val_size (int): Number of samples to use for validation
        transform_train (callable): Transforms for training data
        transform_test (callable): Transforms for test data
    """
    # Create datasets
    train_dataset = CIFAR10Dataset(root_dir, train=True, transform=transform_train)
    test_dataset = CIFAR10Dataset(root_dir, train=False, transform=transform_test)

    # Split training data into train and validation
    train_size = len(train_dataset) - val_size
    train_data, val_data = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size]
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader, test_loader

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#fix_seed(seed=42)

class gcn():
    def __init__(self):
        pass

    def __call__(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        return (x - mean)/(std + 10**(-6))

class ZCAWhitening():
    def __init__(self, epsilon=1e-4, device="cuda"):
        self.epsilon = epsilon
        self.device = device

    def fit(self, images):
        x = images[0][0].reshape(1, -1)
        self.mean = torch.zeros([1, x.size()[1]]).to(self.device)
        con_matrix = torch.zeros([x.size()[1], x.size()[1]]).to(self.device)
        for i in range(len(images)):
            x = images[i][0].reshape(1, -1).to(self.device)
            self.mean += x / len(images)
            con_matrix += torch.mm(x.t(), x) / len(images)
            if i % 10000 == 0:
                print("{0}/{1}".format(i, len(images)))
        self.E, self.V = torch.linalg.eigh(con_matrix)
        self.E = torch.max(self.E, torch.zeros_like(self.E))
        self.ZCA_matrix = torch.mm(torch.mm(self.V, torch.diag((self.E.squeeze()+self.epsilon)**(-0.5))), self.V.t())
        print("completed!")

    def __call__(self, x):
        size = x.size()
        x = x.reshape(1, -1).to(self.device)
        x -= self.mean
        x = torch.mm(x, self.ZCA_matrix.t())
        x = x.reshape(tuple(size))
        x = x.to("cpu")
        return x

# noise makes training a little more stable
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p = 0.5, device="cuda"):
        self.std = std
        self.mean = mean
        self.p = p
        self.device = device

    def __call__(self, tensor):
        rand_num = torch.rand(1).item()
        tensor = tensor.to(self.device)
        if rand_num < self.p:
            noise = torch.randn(tensor.size()).to(self.device)
            return tensor + noise * self.std + self.mean
        else:
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class NormalizeImage():
    def __call__(self, x):
        return torch.div(x, 255)


#test data augumentation
def tta(image_batch, model, device):
    h_flip = transforms.RandomHorizontalFlip(p=1)
    v_flip = transforms.RandomVerticalFlip(p=1)
    crop = transforms.RandomCrop(32, padding=4, padding_mode='reflect')
    noise = AddGaussianNoise(std = 0.03)
    averaged_preds = []
    with torch.no_grad():
        for image in image_batch:
            original_image = image.unsqueeze(0)# Add batch dimension
            original_image.to(device)
            h_flipped_image = h_flip(original_image).to(device)
            v_flipped_image = v_flip(original_image).to(device)
            cropped_images = [crop(original_image).to(device) for _ in range(4)]
            noisy_image = noise(original_image)
            augmented_images = torch.cat([original_image, h_flipped_image, v_flipped_image, noisy_image] + cropped_images, dim=0)
            preds = model(augmented_images)

            avg_pred = preds.mean(dim=0)
            averaged_preds.append(avg_pred)

        averaged_preds_tensor = torch.stack(averaged_preds, dim=0)

    return averaged_preds_tensor

import lightning  as pl
class LitResNet(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.loss_fn = nn.CrossEntropyLoss()


        def training_step(self, batch, batch_idx):
            x, t = batch
            y = self.model(x)
            loss = self.loss_fn(y, t)

            acc = (y.argmax(1) == t).float().mean()
            values = {'loss: ': loss, "acc: ": acc}
            self.log_dict( values, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, t = batch
            y = self.model(x)
            loss = self.loss_fn(y, t)
            acc = (y.argmax(1) == t).float().mean()
            values = {'val_loss: ': loss, "val_acc: ": acc}
            self.log_dict(values, on_epoch=True, prog_bar=True)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)
            return {"optimizer": optimizer,
                    "lr_scheduler":{
                        "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.001),
                        "interval": "epoch"
                    }
                }

def main():
    # optional preprocessing
    # zca = ZCAWhitening()
    # zca.fit(trainval_data)

    transform_train = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
                                    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                    #zca,
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    AddGaussianNoise(std= 0.03)
                                    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    dataloader_train, dataloader_valid, dataloader_test = create_dataloaders(
        root_dir='./data/cifar-10-batches-py',
        batch_size=64,
        val_size=500,
        transform_train=transform_train,
        transform_test=transform_test
    )
    print("preprocessing is done.")

    random_state = 5
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from model import ResNet, BasicBlock

    conv_net = ResNet(BasicBlock, [2, 2, 2, 2])

    """
    def init_weights(m):  # Heの初期化
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.0)

    conv_net.apply(init_weights)
    """
    n_epochs = 200
    lr =  0.01
    momentum = 0.9
    weight_decay = 5e-4
    amp = False

    compile_model = False
    device = 'cuda'
    conv_net.to(device)
    optimizer = optim.SGD(conv_net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0.001)
    loss_function = nn.CrossEntropyLoss()

    print("torch compile enabled :", compile_model)
    if compile_model:
        conv_net = torch.compile(conv_net)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    """
    print('Start lightning training.')
    lit_model = LitResNet(conv_net)
    
    trainer = pl.Trainer(max_epochs=200, 
                         accelerator="gpu", 
                         default_root_dir="D:/coding_practice/DLbasic_hw",
                         gradient_clip_val=5.0,
                         )
    
    trainer.fit(lit_model, dataloader_train, dataloader_valid, )
    
    """

    print('Start training.')
    print("using", device)
    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []
        n_train = 0
        acc_train = 0
        conv_net.train()
        for x, t in tqdm(dataloader_train):
            n_train += t.size()[0]
            conv_net.zero_grad(set_to_none=True)

            x = x.to(device)
            t = t.to(device)

            y = conv_net.forward(x)
            loss = loss_function(y, t)
            loss.backward()

            optimizer.step()
            nn.utils.clip_grad_norm_(conv_net.parameters(), 2.0)

            pred = y.argmax(1)

            acc_train += (pred == t).float().sum().item()
            losses_train.append(loss.tolist())

        scheduler.step()
        with torch.no_grad():
            conv_net.eval()
            n_val = 0
            acc_val = 0
            for x, t in dataloader_valid:
                n_val += t.size()[0]
                x = x.to(device)

                t = t.to(device)

                y = tta(x, conv_net, device)
                loss = loss_function(y, t)
                pred = y.argmax(1)

                acc_val += (pred == t).float().sum().item()
                losses_valid.append(loss.tolist())

            print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
                epoch,
                np.mean(losses_train),
                acc_train/n_train,
                np.mean(losses_valid),
                acc_val/n_val
            ))

    conv_net.eval()
    t_pred = []
    for x in dataloader_test:

        x = x.to(device)

        y = tta(x, conv_net, device)

        pred = y.argmax(1).tolist()

        t_pred.extend(pred)

    torch.save(conv_net.state_dict(), 'trained_models/model.pth')

if __name__ == '__main__':
    main()
