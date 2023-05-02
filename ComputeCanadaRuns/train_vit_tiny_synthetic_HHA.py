ROOT_DATA_DIR = 'data/HHA_Depth_prePT/'
ROOT_TRAIN_VAL_DATA_DIR = 'data/current/'
SAVED_MODELS_DIR = 'savedModels/'
BASE_MODEL_NAME = 'vit_tiny_synthetic_HHA'
TRAIN_DIR = ROOT_TRAIN_VAL_DATA_DIR + 'train/'
VAL_DIR = ROOT_TRAIN_VAL_DATA_DIR + 'val/'
K_FOLDS = 5
BATCH_SIZE = 16
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CLASS_WEIGHT = None
INITIAL_BIAS = None
LEARNING_RATE = 0.00001
MOMENTUM = 0.9
FINETUNING = False

import copy
import os
import shutil
import sys
import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
# from tqdm import tqdm
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torchvision import datasets, models, transforms

runNum = sys.argv[1]
totalRuns = sys.argv[2]
# environment config
os.environ['TORCH_HOME'] = 'torchHome'
# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

noone_p1 = glob(ROOT_DATA_DIR + 'p1/noone/*.png')
nurse_p1 = glob(ROOT_DATA_DIR + 'p1/nurse/*.png')
noone_p2 = glob(ROOT_DATA_DIR + 'p2/noone/*.png')
nurse_p2 = glob(ROOT_DATA_DIR + 'p2/nurse/*.png')
noone_p5 = glob(ROOT_DATA_DIR + 'p5/noone/*.png')
nurse_p5 = glob(ROOT_DATA_DIR + 'p5/nurse/*.png')
noone_p6 = glob(ROOT_DATA_DIR + 'p6/noone/*.png')
nurse_p6 = glob(ROOT_DATA_DIR + 'p6/nurse/*.png')
noone_p8 = glob(ROOT_DATA_DIR + 'p8/noone/*.png')
nurse_p8 = glob(ROOT_DATA_DIR + 'p8/nurse/*.png')
noone_p9 = glob(ROOT_DATA_DIR + 'p9/noone/*.png')
nurse_p9 = glob(ROOT_DATA_DIR + 'p9/nurse/*.png')
noone_p10 = glob(ROOT_DATA_DIR + 'p10/noone/*.png')
nurse_p10 = glob(ROOT_DATA_DIR + 'p10/nurse/*.png')
noone_p11 = glob(ROOT_DATA_DIR + 'p11/noone/*.png')
nurse_p11 = glob(ROOT_DATA_DIR + 'p11/nurse/*.png')
noone_p13 = glob(ROOT_DATA_DIR + 'p13/noone/*.png')
nurse_p13 = glob(ROOT_DATA_DIR + 'p13/nurse/*.png')
noone_p14 = glob(ROOT_DATA_DIR + 'p14/noone/*.png')
nurse_p14 = glob(ROOT_DATA_DIR + 'p14/nurse/*.png')
noone_p15 = glob(ROOT_DATA_DIR + 'p15/noone/*.png')
nurse_p15 = glob(ROOT_DATA_DIR + 'p15/nurse/*.png')
noone_p16 = glob(ROOT_DATA_DIR + 'p16/noone/*.png')
nurse_p16 = glob(ROOT_DATA_DIR + 'p16/nurse/*.png')
noone_p17 = glob(ROOT_DATA_DIR + 'p17/noone/*.png')
nurse_p17 = glob(ROOT_DATA_DIR + 'p17/nurse/*.png')
noone_p21 = glob(ROOT_DATA_DIR + 'p21/noone/*.png')
nurse_p21 = glob(ROOT_DATA_DIR + 'p21/nurse/*.png')
noone_p22 = glob(ROOT_DATA_DIR + 'p22/noone/*.png')
nurse_p22 = glob(ROOT_DATA_DIR + 'p22/nurse/*.png')
noone_p23 = glob(ROOT_DATA_DIR + 'p23/noone/*.png')
nurse_p23 = glob(ROOT_DATA_DIR + 'p23/nurse/*.png')
noone_p24 = glob(ROOT_DATA_DIR + 'p24/noone/*.png')
nurse_p24 = glob(ROOT_DATA_DIR + 'p24/nurse/*.png')
noone_p25 = glob(ROOT_DATA_DIR + 'p25/noone/*.png')
nurse_p25 = glob(ROOT_DATA_DIR + 'p25/nurse/*.png')
noone_p26 = glob(ROOT_DATA_DIR + 'p26/noone/*.png')
nurse_p26 = glob(ROOT_DATA_DIR + 'p26/nurse/*.png')
noone_p27 = glob(ROOT_DATA_DIR + 'p27/noone/*.png')
nurse_p27 = glob(ROOT_DATA_DIR + 'p27/nurse/*.png')
noone_p28 = glob(ROOT_DATA_DIR + 'p28/noone/*.png')
nurse_p28 = glob(ROOT_DATA_DIR + 'p28/nurse/*.png')
noone_p29 = glob(ROOT_DATA_DIR + 'p29/noone/*.png')
nurse_p29 = glob(ROOT_DATA_DIR + 'p29/nurse/*.png')
noone_p30 = glob(ROOT_DATA_DIR + 'p30/noone/*.png')
nurse_p30 = glob(ROOT_DATA_DIR + 'p30/nurse/*.png')
noone_p31 = glob(ROOT_DATA_DIR + 'p31/noone/*.png')
nurse_p31 = glob(ROOT_DATA_DIR + 'p31/nurse/*.png')
noone_p32 = glob(ROOT_DATA_DIR + 'p32/noone/*.png')
nurse_p32 = glob(ROOT_DATA_DIR + 'p32/nurse/*.png')
noone_p33 = glob(ROOT_DATA_DIR + 'p33/noone/*.png')
nurse_p33 = glob(ROOT_DATA_DIR + 'p33/nurse/*.png')
noone_p34 = glob(ROOT_DATA_DIR + 'p34/noone/*.png')
nurse_p34 = glob(ROOT_DATA_DIR + 'p34/nurse/*.png')
nurse_p90 = glob(ROOT_DATA_DIR + 'p90/nurse/*.png')
nurse_p91 = glob(ROOT_DATA_DIR + 'p91/nurse/*.png')
nurse_p92 = glob(ROOT_DATA_DIR + 'p92/nurse/*.png')
nurse_p93 = glob(ROOT_DATA_DIR + 'p93/nurse/*.png')
nurse_p94 = glob(ROOT_DATA_DIR + 'p94/nurse/*.png')


all_noone_list = [noone_p1, noone_p2, noone_p5, noone_p6, noone_p8, noone_p9, noone_p10, noone_p11, noone_p13, noone_p14, noone_p15, noone_p16, noone_p17, noone_p21, noone_p22, noone_p23, noone_p24, noone_p25, noone_p26, noone_p27, noone_p28, noone_p29, noone_p30, noone_p31, noone_p32, noone_p33, noone_p34, [], [], [], [], []]
all_nurse_list = [nurse_p1, nurse_p2, nurse_p5, nurse_p6, nurse_p8, nurse_p9, nurse_p10, nurse_p11, nurse_p13, nurse_p14, nurse_p15, nurse_p16, nurse_p17, nurse_p21, nurse_p22, nurse_p23, nurse_p24, nurse_p25, nurse_p26, nurse_p27, nurse_p28, nurse_p29, nurse_p30, nurse_p31, nurse_p32, nurse_p33, nurse_p34, nurse_p90, nurse_p91, nurse_p92, nurse_p93, nurse_p94]

def prepFiles(noone_train, nurse_train, noone_val, nurse_val):
    # Clean up train and val folders
    remFiles = glob(TRAIN_DIR + 'noone/'+ '*.png')
    for f in remFiles:
        os.remove(f)
    remFiles = glob(TRAIN_DIR + 'nurse/'+ '*.png')
    for f in remFiles:
        os.remove(f)
    remFiles = glob(VAL_DIR + 'noone/'+ '*.png')
    for f in remFiles:
        os.remove(f)
    remFiles = glob(VAL_DIR + 'nurse/'+ '*.png')
    for f in remFiles:
        os.remove(f)

    # Move files of current fold into their folders
    for f in noone_train:
        basename = os.path.basename(f)
        dst_path = TRAIN_DIR + 'noone/' + basename
        shutil.copy(f, dst_path)
    for f in nurse_train:
        basename = os.path.basename(f)
        dst_path = TRAIN_DIR + 'nurse/' + basename
        shutil.copy(f, dst_path)
    for f in noone_val:
        basename = os.path.basename(f)
        dst_path = VAL_DIR + 'noone/' + basename
        shutil.copy(f, dst_path)
    for f in nurse_val:
        basename = os.path.basename(f)
        dst_path = VAL_DIR + 'nurse/' + basename
        shutil.copy(f, dst_path)

    noone_train = glob(TRAIN_DIR + 'noone/*.png')
    nurse_train = glob(TRAIN_DIR + 'nurse/*.png')

    noone_val = glob(VAL_DIR + 'noone/*.png')
    nurse_val = glob(VAL_DIR + 'nurse/*.png')

    return noone_train, nurse_train, noone_val, nurse_val

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler = None, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-'*10)

        # Each epoch trains and validates
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if in 'train'
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward & optimize only if in 'train'
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                del inputs, labels, outputs, loss

            if scheduler and phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def create_model(finetuning=True):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)


    for param in model.parameters():
        param.requires_grad = not finetuning

    new_head = nn.Linear(model.head.in_features, 2, )
    new_head.weight.data.fill_(0)
    new_head.bias.data.fill_(0)
    model.head = new_head
    
    return model

def printCM(cm, labels):
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    # ax.xaxis.set_ticklabels(labels)
    # ax.yaxis.set_ticklabels(labels)
    return

def evalModel(model, validation_dataloader):
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in validation_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            predicted = predicted.data.cpu().numpy()

            labels = labels.data.cpu().numpy()

            y_pred.extend(predicted)
            y_true.extend(labels)

    classes = ('noone', 'nurse')

    cf_matrix = confusion_matrix(y_true, y_pred)
    printCM(cf_matrix, labels)

    TN = cf_matrix[0][0]
    FP = cf_matrix[0][1]
    FN = cf_matrix[1][0]
    TP = cf_matrix[1][1]
    prec = TP / (TP + FP)
    spec = TN / (TN + FP)
    sens = TP / (TP + FN)
    acc = (TP + TN) / (TP + TN + FP + FN)
    f1 = (2 * TP) / ((2 * TP) + FP + FN)
    print(f'TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}')
    print(f"Sens: {sens}, Spec: {spec}, Prec: {prec}, Acc: {acc}, F1: {f1}\n")

data_transforms = {
    'train': transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(), # also rescales
    ]),
    'val': transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        # transforms.RandomRotation(180),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(), # also rescales
    ])
}

all_train_idx = np.array([
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ])
all_val_idx = (all_train_idx == 0).astype('int32')

for currRun in range(int(totalRuns)):
    foldNum = 0
    runSaveNum = int(runNum) + int(currRun)
    for train_index, val_index in zip(all_train_idx, all_val_idx):
        train_idx = train_index.nonzero()[0]
        val_idx = val_index.nonzero()[0]

        print(f'Fold: {foldNum}')

        # Select the current folds for training and validation
        noone_train = []
        nurse_train = []
        noone_val = []
        nurse_val = []
        for i in train_idx:
            noone_train += all_noone_list[i]
            nurse_train += all_nurse_list[i]
        for i in val_idx:
            noone_val += all_noone_list[i]
            nurse_val += all_nurse_list[i]

        # Move images to training and testing folders
        noone_train, nurse_train, noone_val, nurse_val = prepFiles(noone_train, nurse_train, noone_val, nurse_val)

        print(f"noone_train: {len(noone_train)}, nurse_train: {len(nurse_train)}")
        print(f"noone_val: {len(noone_val)}, nurse_val: {len(nurse_val)}\n")

        weight_for_0 = (1 / (len(noone_train) + len(noone_val))) * ((len(nurse_train) + len(nurse_val) + len(noone_train) + len(noone_val)) / 2.0)
        weight_for_1 = (1 / (len(nurse_train) + len(nurse_val))) * ((len(nurse_train) + len(nurse_val) + len(noone_train) + len(noone_val)) / 2.0)
        CLASS_WEIGHT = torch.tensor([weight_for_0, weight_for_1], device=device)

        # import data
        sets = ['train', 'val']
        image_datasets = {x: datasets.ImageFolder(os.path.join(ROOT_TRAIN_VAL_DATA_DIR, x), data_transforms[x]) for x in sets}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in sets}
        dataset_sizes = {x: len(image_datasets[x]) for x in sets}

        class_names = image_datasets['train'].classes
        print(class_names)

        ###
        # train everything

        model = create_model(finetuning=FINETUNING)

        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


        model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=20)

        foldNum += 1

        modelPath = SAVED_MODELS_DIR + BASE_MODEL_NAME + "_run" + str(runSaveNum) + "_fold" + str(foldNum) + ".pth"
        torch.save(model.state_dict(), modelPath)

        evalModel(model, dataloaders['val'])