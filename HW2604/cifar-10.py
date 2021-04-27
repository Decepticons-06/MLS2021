import torch
import numpy as np
import pandas as pd
from PIL import Image
from imutils import paths
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

! git clone https://github.com/YoongiKim/CIFAR-10-images

# mounting google drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

# getting all the image paths
train_path = '/content/CIFAR-10-images/train'
train_image_paths = list(paths.list_images(train_path))

test_path = '/content/CIFAR-10-images/test'
test_image_paths = list(paths.list_images(test_path))

# creating a dataframe for test_data contains all image paths and corresponding labels
test_data = pd.DataFrame(columns=['image_path', 'target'])
test_labels = []

for i, image_path in enumerate(test_image_paths):
    test_data.loc[i, 'image_path'] = image_path
    test_label = image_path[len(test_path):].split('/')[1]
    test_labels.append(test_label)

test_labels = np.array(test_labels)
# one-hot encoding
labels = LabelBinarizer().fit_transform(test_labels)

for i in range(len(labels)):
    index = np.argmax(labels[i])
    test_data.loc[i,"target"] = index

# creating a dataframe for train_data contains all image paths and corresponding labels
train_data = pd.DataFrame(columns=['image_path', 'target'])
train_labels = []

for i, image_path in enumerate(train_image_paths):
    train_data.loc[i, 'image_path'] = image_path
    train_label = image_path[len(test_path):].split('/')[1]
    train_labels.append(train_label)

train_labels = np.array(train_labels)
# one-hot encoding
labels = LabelBinarizer().fit_transform(train_labels)

for i in range(len(labels)):
    index = np.argmax(labels[i])
    train_data.loc[i,"target"] = index

# creating csv file from the dataframe
train_data = train_data.sample(frac=1).reset_index(drop=True) # shuffle the dataset
train_data.to_csv(train_path+'/train.csv', index=False)

test_data = test_data.sample(frac=1).reset_index(drop=True)
test_data.to_csv(test_path+'/test.csv', index=False)

# creating train and valid samplers
valid_size = 0.2

num_train = len(train_data)
indices = list(range(num_train))
split = int(valid_size * num_train)
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# creating dataset module
class CIFAR10Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.X = self.df.image_path.values
        self.y = self.df.target.values
        self.transform = transform

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, idx):
        image = Image.open(self.X[idx])
        image = self.transform(image)
        label = self.y[idx]

        return image, label

# defning train and test transformations
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                           (0.2023, 0.1994, 0.2010)),
                                      ])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.2010)),
                                    ])

# creating iterable data loaders
train_csv_path = train_path+'/train.csv'
test_csv_path = test_path+'/test.csv'

train_data = CIFAR10Dataset(train_csv_path, train_transform)
test_data = CIFAR10Dataset(test_csv_path, test_transform)

batch = 20

train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch)
valid_loader = DataLoader(train_data, sampler=valid_sampler, batch_size=batch)
test_loader = DataLoader(test_data, batch_size=batch)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# sample training dataset
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize=(25, 4))

def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

for idx in np.arange(batch):
    ax = fig.add_subplot(2, batch/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

# sample test dataset
dataiter = iter(test_loader)
images, labels = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize=(25, 4))

def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

for idx in np.arange(batch):
    ax = fig.add_subplot(2, batch/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

# importing cifar-10_cnn model from drive
import sys
sys.path.append('/content/gdrive/My Drive/Colab Notebooks')

import cifar10_cnn

# defining model, criterion, optimizer
model = cifar10_cnn.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# placing the model to gpu
model.cuda()

# training with validation
def train(save_name, trainloader, validloader, criterion, optimizer, epochs):
    valid_loss_min = np.Inf

    for e in range(epochs):
        train_loss = 0
        valid_loss = 0

        model.train()
        for images, labels in trainloader:
            images, labels = images.cuda(), labels.cuda()
            log_probs = model(images)
            loss = criterion(log_probs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*len(images)
        else:
            model.eval()
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.cuda(), labels.cuda()
                    log_probs = model(images)
                    loss = criterion(log_probs, labels)
                    valid_loss += loss.item()*len(images)

        train_loss = train_loss/len(trainloader.sampler)
        valid_loss = valid_loss/len(validloader.sampler)

        print("epoch: {:2d}/{}".format(e+1, epochs),
              "train_loss: {:.3f}".format(train_loss),
              "valid_loss: {:.3f}".format(valid_loss))

        # saving the model
        if valid_loss <= valid_loss_min:
            path = F"/content/{save_name}"
            torch.save(model.state_dict(), path)
            valid_loss_min = valid_loss

# testing
def test(model_file_name, test_loader, criterion):
    path = F"/content/{model_file_name}"
    model.load_state_dict(torch.load(path))

    test_loss = 0
    test_accuracy = 0

    model.eval()
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        output = model(images)
        loss = criterion(output, labels)
        test_loss += loss.item()
        probs = torch.exp(output)
        top_prob, top_class = probs.topk(1, dim=1)
        equals = top_class == labels.view(top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    test_loss = test_loss/len(test_loader)
    test_accuracy = test_accuracy/len(test_loader)

    print('Test Loss: {:.3f} '.format(test_loss),
          'Test Accuracy: {:.3f}'.format(test_accuracy))

    fp = open("/content/gdrive/My Drive/Colab Notebooks/performance.txt", "x")
    fp.write('Test Loss: {:.3f}... Test Accuracy: {:.3f}'.format(test_loss, test_accuracy))
    fp.close()

def main():
    epochs = 50
    save_name = 'cifar10_model'

    train(save_name, train_loader, valid_loader, criterion, optimizer, epochs)
    test(save_name, test_loader, criterion)

if __name__ == "__main__":
    main()
