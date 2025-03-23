import sys
import os
import glob
import random
import time
import copy
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

#--------------------------------------------------------------------------------------------
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, img_labels,transform = None, target_transform = None):
        self.img_list = img_list
        self.img_labels = img_labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        label = labels_dict[self.img_labels[idx]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label

#--------------------------------------------------------------------------------------------

def collate_fn(batch):
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor,labels_tensor

#--------------------------------------------------------------------------------------------

def train_model(model, criterion, optimizer, scheduler,dataloaders,device,dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    loss_history={
        "train": [],
        "val": []
    }
    
    metric_history={
        "train": [],
        "val": []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}',flush = True)
        print('-' * 10,flush = True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            t0 = time.perf_counter()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                loss_history["train"].append(epoch_loss)
                metric_history["train"].append(epoch_acc)
            else:
                loss_history["val"].append(epoch_loss)
                metric_history["val"].append(epoch_acc)
                
            t1 = time.perf_counter()-t0

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {t1:.4f}',flush = True)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s',flush = True)
    print(f'Best val Acc: {best_acc:4f}',flush = True)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,loss_history,metric_history

#--------------------------------------------------------------------------------------------


def main(image_dir,output_name,num_epochs,num_sample):

    cudnn.benchmark = True

    os.makedirs(output_name, exist_ok=True)

    catgs = os.listdir(image_dir)
    if '.DS_Store' in catgs:
        catgs.remove(".DS_Store")

    global labels_dict
    labels_dict = {}
    ind = 0
    for uc in catgs:
        labels_dict[uc] = ind
        ind+=1
        
    print(f"labels_dict: {labels_dict}",flush = True)

    if ind > 2:
        print("More than 2 directories found. Only directories named PILING or NONPILING will be used. Remaining directories will be ignored.")

    if ind < 2:
        print("Less than 2 directories found. Binary classification requires 2 classes. Directories for each class should be clearly labelled as either PILING or NONPILING.")


    filenamesPILING = glob.glob(f"{image_dir}/PILING/*.jpg")
    filenamesNONPILING = glob.glob(f"{image_dir}/NONPILING/*.jpg")

    # Set a random seed for reproducibility
    random.seed(2024)

    idx = random.sample(list(np.arange(len(filenamesPILING))),num_sample)
    jdx = random.sample(list(np.arange(len(filenamesNONPILING))),num_sample)

    # Save the random indexes so we can filter them out from further testing
    random_index = {'idx':idx,'jdx' : jdx}
    with open(os.path.join(output_name,f"model-index-samples-{num_sample}.dat"),'wb') as f:
        pickle.dump(random_index,f)

    unique_ids =  [filenamesPILING[i] for i in idx] + [filenamesNONPILING[j] for j in jdx]
    unique_labels = ['PILING']*len(idx) + ['NONPILING']*len(jdx)

    # Split into training and validation datasetd
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=1)
    train_indx, test_indx = next(sss.split(unique_ids, unique_labels))

    train_ids = [unique_ids[ind] for ind in train_indx]
    train_labels = [unique_labels[ind] for ind in train_indx]
    print(len(train_ids), len(train_labels),flush = True)
    
    test_ids = [unique_ids[ind] for ind in test_indx]
    test_labels = [unique_labels[ind] for ind in test_indx]
    print(len(test_ids), len(test_labels),flush = True)


    # Change this to the dimensions of the input training data
    w, h = 672, 295
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Original image size and post-crop dimensions. Standardise images and crop to center before applying other transformations
    ow, oh = 2688, 1520
    cw, ch = 2688, 1180

    train_transformer = transforms.Compose([
        transforms.Resize((oh,ow)),
        transforms.CenterCrop((ch,cw)),
        transforms.Grayscale(num_output_channels = 3),
        transforms.Resize((h,w)),
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),    
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    test_transformer = transforms.Compose([
        transforms.Resize((oh,ow)),
        transforms.CenterCrop((ch,cw)),
        transforms.Grayscale(num_output_channels = 3),
        transforms.Resize((h,w)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])


    train_ds = MyDataset(train_ids,train_labels,train_transformer)
    test_ds = MyDataset(test_ids,test_labels,test_transformer)


    batch_size = 16
    train_dl = DataLoader(train_ds, batch_size= batch_size,shuffle=True,collate_fn = collate_fn)
    test_dl = DataLoader(test_ds, batch_size= 2*batch_size,shuffle=False,collate_fn = collate_fn)

    dataloaders = {'train' : train_dl,'val' : test_dl}

    dataset_sizes = {'train' : len(train_ds), 'val' : len(test_ds)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Train the model
    model_ft,loss_history,metric_history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,dataloaders,device, dataset_sizes, num_epochs)

    # Save model
    model_path = os.path.join(output_name,'model-state-dict.dat')
    torch.save(model_ft.state_dict(), model_path)

    # Save model loss for later plotting/inspection
    with open(os.path.join(output_name,'model-loss-hist.dat'),'wb') as f:
        pickle.dump(loss_history,f)

    # Save Figure of Model Loss
    plt.title("Train-Val Loss")
    plt.plot(range(1,num_epochs+1),loss_history["train"],label="Training")
    plt.plot(range(1,num_epochs+1),loss_history["val"],label="Validation")
    plt.xticks(range(0,num_epochs+1,2))
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig('TrainVal_Loss.png', dpi=300)
    plt.clf()    

    # Save model metric for later plotting/inspection
    with open(os.path.join(output_name,'model-metric-hist.dat'),'wb') as f:
        pickle.dump(metric_history,f)

    train_acc = [acc.cpu().item() for acc in metric_history["train"]]
    val_acc = [acc.cpu().item() for acc in metric_history["val"]]

    # Save Figure of Model Accuracy
    plt.title("Train-Val Accuracy")
    plt.plot(range(1,num_epochs+1), train_acc,label="Training")
    plt.plot(range(1,num_epochs+1), val_acc,label="Validation")
    plt.xticks(range(0,num_epochs+1,2))
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig('TrainValAcc.png', dpi=300)
    plt.clf()

    # Save Figure both Accuracy and Loss on same axes
    plt.title("Train-Val Accuracy & Loss")
    plt.plot(range(1,num_epochs+1),loss_history["train"],label="Training Loss")
    plt.plot(range(1,num_epochs+1),loss_history["val"],label="Validation Loss")
    plt.plot(range(1,num_epochs+1), train_acc,label="Training Accuracy")
    plt.plot(range(1,num_epochs+1), val_acc,label="Validation Accuracy")
    plt.xticks(range(0,num_epochs+1,2))
    plt.ylabel("Accuracy / Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig('TrainValAccLoss.png', dpi=300)
    plt.clf()

#--------------------------------------------------------------------------------------------

if __name__ == '__main__':

    if len(sys.argv) == 1 or len(sys.argv) > 5:
        print(f"[!] Please provide arguments: IMAGE_DIR, OUTPUT_NAME (OPTIONAL), NUM_EPOCHS (OPTIONAL), NUM_SAMPLES (OPTIONAL")
    else:
        print(sys.argv[1])
        image_dir = os.path.abspath(sys.argv[1])
        print(image_dir)
        if len(sys.argv) == 5:    
            output_name = os.path.abspath(sys.argv[2])
            num_epochs = int(sys.argv[3])
            num_sample = int(sys.argv[4])
        elif len(sys.argv) == 4:
            output_name = os.path.abspath(sys.argv[2])
            num_epochs = int(sys.argv[3])
            num_sample = 5000
        elif len(sys.argv) == 3:
            output_name = os.path.abspath(sys.argv[2])
            num_epochs = 20
            num_sample = 5000
        elif len(sys.argv) == 2:
            output_name = os.path.abspath("Output")
            num_epochs = 20
            num_sample = 5000
        
        main(image_dir,output_name,num_epochs,num_sample)