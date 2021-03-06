'''
This is starter code for Assignment 2 Problem 1 of CMPT 726 Fall 2020.
The file is adapted from the repo https://github.com/chenyaofo/CIFAR-pretrained-models
'''

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt

NUM_EPOCH = 10


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


######################################################
####### Do not modify the code above this line #######
######################################################

class cifar_resnet20(nn.Module):
    def __init__(self):
        super(cifar_resnet20, self).__init__()
        ResNet20 = CifarResNet(BasicBlock, [3, 3, 3])
        url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt'

        ResNet20.load_state_dict(model_zoo.load_url(url))
        modules = list(ResNet20.children())[:-1]
        backbone = nn.Sequential(*modules)
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.shape[0], -1)
        return self.fc(out)


if __name__ == '__main__':
    PATH = './cifar_model.pth'
    model = cifar_resnet20()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                         std=(0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    total_training_loss = []
    total_training_accuracy = []

    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)
    total_testing_loss = []
    total_testing_accuracy = []
    min_loss = 0
    min_epoch = 0

    criterion = nn.CrossEntropyLoss()

    # Try applying L2 regularization to the coefficients in the small networks we added.
    optimizer = optim.SGD(list(model.fc.parameters()), lr=0.001, momentum=0.9, weight_decay=0.00001)

    #  Do the training
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        train_match = 0
        total_training_data = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            train_match += (predict == labels).sum().item()
            total_training_data += labels.size(0)
            if i % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d] training loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
            training_loss = running_loss / len(trainloader)
            training_accuracy = train_match / total_training_data

        total_training_loss.append(training_loss)
        total_training_accuracy.append(training_accuracy)

        # Do the testing
        testing_running_loss = 0.0
        test_match = 0
        total_testing_data = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                testing_loss = criterion(outputs, labels)
                testing_running_loss += testing_loss.item()
                _, predict = torch.max(outputs.data, 1)
                test_match += (predict == labels).sum().item()
                total_testing_data += labels.size(0)
                if i % 20 == 19:  # print every 20 mini-batches
                    print('[%d, %5d] testing loss: %.3f' %
                          (epoch + 1, i + 1, testing_running_loss / 20))
                    testing_running_loss = 0.0
            testing_loss = testing_running_loss / len(testloader)
            testing_accuracy = test_match / total_testing_data
            if min_loss == 0:
                min_loss = testing_loss
            elif testing_loss < min_loss:
                min_loss = testing_loss
                min_epoch = epoch
                best_model_state = model.state_dict()
                torch.save(best_model_state, PATH)

        total_testing_loss.append(testing_loss)
        total_testing_accuracy.append(testing_accuracy)



    print('Finished Training')
    print('**********Model Result**********')
    min_training_loss = min(total_training_loss)
    print("Minimum Training Loss:", min_training_loss)
    print("Training Accuracy:", training_accuracy)
    min_testing_loss = min(total_testing_loss)
    print('Minimum loss in epoch:', min_epoch + 1)
    print("Minimum Testing Loss:", min_testing_loss)
    print("Testing Accuracy:", testing_accuracy)

    # Prediction
    # load the trained model

    classes = ('airplanes', 'cars', 'birds', 'cats',
               'deers', 'dogs', 'frogs', 'horses', 'ships', 'trucks')
    model = cifar_resnet20()
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # one image input
    test_img = testset[0]
    test_output = model(test_img[0].reshape(1, 3, 32, 32))
    _, prediction = torch.max(test_output, dim=1)
    print('**********Prediction Result of 1 image**********')
    print("Expected result: " + classes[test_img[1]])
    print("Prediction from the model: " + classes[prediction])
    plt.imshow(np.transpose(test_img[0].numpy(), (1, 2, 0)))
    plt.show()
