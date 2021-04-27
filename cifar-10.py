! git clone https://github.com/YoongiKim/CIFAR-10-images

from imutils import paths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# getting all the image paths
train_path = '/content/CIFAR-10-images/train'
train_image_paths = list(paths.list_images(train_path))

test_path = '/content/CIFAR-10-images/test'
test_image_paths = list(paths.list_images(test_path))

# creating a dataframe contains all image paths and corresponding labels/classes
test_data = pd.DataFrame(columns=['image_path', 'target'])
test_labels = []

train_data = pd.DataFrame(columns=['image_path', 'target'])
train_labels = []

for i, image_path in enumerate(test_image_paths):
    test_data.loc[i, 'image_path'] = image_path
    test_label = image_path[len(test_path):].split('/')[1]
    test_labels.append(test_label)

test_data['target'] = test_labels

for i, image_path in enumerate(train_image_paths):
    train_data.loc[i, 'image_path'] = image_path
    train_label = image_path[len(test_path):].split('/')[1]
    train_labels.append(train_label)

train_data['target'] = train_labels

# creating a csv file from the dataframe
train_data = train_data.sample(frac=1).reset_index(drop=True) #shuffle the dataset
train_data.to_csv(train_path+'/train.csv', index=False)

test_data = test_data.sample(frac=1).reset_index(drop=True) #shuffle the dataset
test_data.to_csv(test_path+'/test.csv', index=False)

# reading the csv file and dividing into train, test file
df = pd.read_csv(train_path+'/train.csv')
X = df.image_path.values
y = df.target.values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# creating dataset module
class CIFAR10Dataset(Dataset):
    def __init__(self, X, y, validate=False):
        self.X = X
        self.y = y

        if validate:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                                      (0.5, 0.5, 0.5))
                                                 ])
        else:
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(10),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                                      (0.5, 0.5, 0.5))
                                                 ])

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, idx):
        image = Image.open(self.X[idx])
        image = self.transform(image)
        label = self.y[idx]

        return image, label

# creating iterable data loaders
train_data = CIFAR10Dataset(X_train, y_train)
valid_data = CIFAR10Dataset(X_valid, y_valid, validate=True)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# sample training dataset
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize=(25, 4))

def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(labels[idx])
