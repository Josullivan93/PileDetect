import sys
import os
import argparse
import glob
import random
import csv
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, img_labels=None ,transform = None, is_labeled=True, labels_dict=None):
        self.img_list = img_list
        self.img_labels = img_labels
        self.transform = transform
        self.is_labeled = is_labeled
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        if self.is_labeled:
            label = self.labels_dict[self.img_labels[idx]]
            return image, label
        else:
            return image, img_path

#--------------------------------------------------------------------------------------------

def collate_fn(batch):
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor,labels_tensor

#--------------------------------------------------------------------------------------------
# Inference collate
def collate_fn_inf(batch):
    imgs_batch, path_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    path_batch = [p for p, imgs in zip(path_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    return imgs_tensor,path_batch

#--------------------------------------------------------------------------------------------

def denormalize(x_, mean, std):
    x = x_.clone()
    for i in range(3):
        x[i] = x[i]*std[i]+mean[i]
    x_pil = to_pil_image(x)
    x_np = np.array(x_pil)

    return x_np

#--------------------------------------------------------------------------------------------

def main(args):

    # Parse required args
    if not os.path.isdir(os.path.abspath(args.directory)):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return
    

    print(args.model_dat)
    print(os.path.abspath(args.model_dat))
    if not os.path.isfile(os.path.abspath(args.model_dat)):
        print(f"Error: Model file '{args.model_dat}' does not exist.")
        return

    image_dir = os.path.abspath(args.directory)
    model_dat = os.path.abspath(args.model_dat)

    print(f"Using directory: {image_dir}")
    print(f"Using model file: {os.path.abspath(args.model_dat)}")


    cudnn.benchmark = True

    # Validation
    # Parse Optional Validation Args
    if args.validate:
        print("Performing Validation.")

        if args.nsamples is not None:
            val_sample = args.nsamples
            print(f"Using {val_sample} samples per class.")
        else:
            val_sample = 2500
            print("No value given for number of validation samples. Using 2500 as default.")

        # Locate and Load data - Take random sample of size val_sample

        filenamesPILING = random.sample(glob.glob(f"{image_dir}/PILING/*.jpg"), val_sample)
        filenamesNONPILING = random.sample(glob.glob(f"{image_dir}/NONPILING/*.jpg"), val_sample)

        unique_ids = filenamesPILING + filenamesNONPILING
        unique_labels = ['PILING']*len(filenamesPILING) + ['NONPILING']*len(filenamesNONPILING)

        # Shuffle the order of images
        img_list = list(zip(unique_ids, unique_labels))
        random.shuffle(img_list)
        unique_ids, unique_labels = zip(*img_list)

        # Create dictionary of labels
        labels_dict = {}
        labels_dict['NONPILING'] = 0
        labels_dict['PILING'] = 1

        print(f"labels_dict: {labels_dict}")

        # Transform by resizing, converting to grayscale and normalising
        # This is the same image processing that we perform on the training examples
        w, h = 672, 295
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Original image size and post-crop dimensions. Standardise images and crop to center before applying other transformations
        ow, oh = 2688, 1520
        cw, ch = 2688, 1180

        deploy_transformer = transforms.Compose([
        transforms.Resize((oh,ow)),
        transforms.CenterCrop((ch,cw)),
        transforms.Grayscale(num_output_channels = 3),
        transforms.Resize((h,w)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

        deploy_ds = MyDataset(unique_ids,unique_labels,deploy_transformer, is_labeled=True, labels_dict=labels_dict)

        # Shuffle off to allow ease of matching File names to Output Labels
        # 10% batch of default
        batch_size = 250
        deploy_dl = DataLoader(deploy_ds, batch_size=batch_size,shuffle=False,collate_fn = collate_fn)  
        dataloaders = {'deploy' : deploy_dl}

        dataset_sizes = {'deploy' : len(deploy_ds)}
        class_names = ['NONPILING','PILING']

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load model
        model_ft = models.resnet18()

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)

        model_ft.load_state_dict(torch.load(model_dat, weights_only = True))

        # Use the evaluation/inference mode of the model
        model_ft.eval()
        
        # Perform Inference
        out_true = []
        out_pred = []
        out_prob = []

        for inputs, labels in tqdm(dataloaders['deploy']):

            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)

                model_ft = model_ft.to(device)

                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                probabilities = F.softmax(outputs, dim=1)[:, 1]

            out_true.extend(labels.tolist())
            out_pred.extend(preds.tolist())
            out_prob.extend(probabilities.tolist())

        # Plot model results as ROC
        # Generate No Skill Prediction Probs
        ns_prob = [0 for _ in range(len(out_prob))]

        ns_fpr, ns_tpr, ns_thresholds = roc_curve(out_true, ns_prob)
        fpr, tpr, thresholds = roc_curve(out_true, out_prob)

        # Calculate AUC
        auc_out = roc_auc_score(out_true, out_prob)

        # plt ROC curves
        plt.title("Receiver Operating Characteristics Curve")
        plt.plot(fpr, tpr, color = "#1A85FF", label = 'RESNET18')
        plt.plot(ns_fpr, ns_tpr, linestyle='--', color = "#D41159", label = 'No Skill')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.text(0, 1, f'AUC = {auc_out:.3f}', fontsize = 10)
        plt.legend()
        plt.savefig('ROC_Curve.png')
        plt.clf()

        print('AUC: ', auc_out)

        # Plot model results as Precision-Recall
        out_precision, out_recall, _ = precision_recall_curve(out_true, out_prob)
        pr_auc_out = auc(out_recall, out_precision)
        
        # plt ROC curves
        plt.title("Precision-Recall Curve")
        plt.plot(out_recall, out_precision, color = "#1A85FF", label = 'RESNET18')
        plt.plot([0, 1], [0.5, 0.5], linestyle='--', color = "#D41159", label = 'No Skill')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.text(0.8, 1, f'AUC = {pr_auc_out:.3f}', fontsize = 10)
        plt.legend()
        plt.savefig('Precision_Recall_Curve.png')
        plt.clf()

        print('PR AUC: ', pr_auc_out)  

        # Calculate & Print Additional Performance Metrics
        print("Device Used: ",device)

        tp_out = sum([a == 1 and a == b for a,b in zip(out_pred, out_true)])
        print("TP:", tp_out)
        tn_out = sum([a == 0 and a == b for a,b in zip(out_pred, out_true)])
        print("TN:", tn_out)

        print("True Sum:", sum([a == b for a,b in zip(out_pred, out_true)]))

        fp_out = sum([a == 1 and a != b for a,b in zip(out_pred, out_true)])
        print("FP:", fp_out)
        fn_out = sum([a == 0 and a != b for a,b in zip(out_pred, out_true)])
        print("FN:", fn_out)

        print("False Sum:", sum([a != b for a,b in zip(out_pred, out_true)]))

        acc_out = sum([a == b for a,b in zip(out_pred, out_true)])/len(out_pred)
        print("Accuracy:", acc_out)

        # Precision (PPV)
        prec_out = tp_out / (tp_out + fp_out)
        print("Precision:", prec_out)

        # Sensitivity / Recall
        rec_out = tp_out / (tp_out + fn_out)
        print("Recall:", rec_out)

        # Specificity
        spec_out = tn_out / (tn_out + fp_out)
        print("Specificity:", spec_out)

        # F measure
        f1_out = (2 * prec_out * rec_out)/(prec_out + rec_out)
        print("F1 Measure:", f1_out)

        # Save metrics to file
        output_values = [
            ['Accuracy', acc_out],
            ['Precision', prec_out],
            ['Recall', rec_out],
            ['Specificity', spec_out],
            ['F1 Measure', f1_out],
            ['True Positive', tp_out],
            ['True Negative', tn_out],
            ['False Positive', fp_out],
            ['False Negative', fn_out],
            ['ROC AUC', auc_out],
            ['Precision-Recall AUC', pr_auc_out]
        ]

        with open ('Performance_Metrics.csv','w',newline = '') as csvfile:
            my_writer = csv.writer(csvfile, delimiter = ',')
            my_writer.writerows(output_values)

        # Save list of file paths, true labels, predicted labels

        with open ('Classification_list.csv','w',newline='') as csvfile:
            my_writer = csv.writer(csvfile, delimiter = ',')
            my_writer.writerow(['File Path', 'True Label', 'Classification'])

            for i in range(len(out_pred)):
                img_path = dataloaders['deploy'].dataset.img_list[i]
                truth = dataloaders['deploy'].dataset.img_labels[i]
                prediction = out_pred[i]

                my_writer.writerow([img_path, truth, prediction])
        
        if args.misclass:
            # Create Figures for visual inspection

            print("Creating Misclassification Figures")

            # False Positive
            fp_ind = [i for i, x in enumerate([a == 1 and a != b for a,b in zip(out_pred, out_true)]) if x]
            batch_num = 0
            rep_num = 0
            i = 0

            for test_input, test_labels in dataloaders['deploy']:
                batch_ind = [y for y in [x - (batch_num * batch_size) for x in fp_ind] if y < 100 and y >= 0]
                if len(batch_ind) != 0:
                    temp_batch_ind = random.choice(batch_ind)
                    plt.subplot(5,4,i+1)
                    plt.imshow(denormalize(test_input.cpu().data[temp_batch_ind], mean, std))
                    plt.tick_params(left = False, right = False , labelleft = False , 
                        labelbottom = False, bottom = False) 
                    plt.title(f'Predicted: {class_names[out_pred[temp_batch_ind+(batch_num * batch_size)]]}, Actual: {class_names[out_true[temp_batch_ind+(batch_num * batch_size)]]} \n Prediction Probability: {out_prob[temp_batch_ind+(batch_num * batch_size)]:.2f}')
                    i+=1
                batch_num+=1

                if rep_num == 0 and batch_num > 18:
                    batch_num = random.choice(list(range(0,18)))
                    batch_ind = [y for y in [x - (batch_num * batch_size) for x in fp_ind] if y < 100 and y >= 0]
                    if len(batch_ind) != 0:
                        temp_batch_ind = random.choice(batch_ind)
                        plt.subplot(5,4,i+1)
                        plt.imshow(denormalize(test_input.cpu().data[temp_batch_ind], mean, std))
                        plt.tick_params(left = False, right = False , labelleft = False , 
                        labelbottom = False, bottom = False) 
                        plt.title(f'Predicted: {class_names[out_pred[temp_batch_ind+(batch_num * batch_size)]]}, Actual: {class_names[out_true[temp_batch_ind+(batch_num * batch_size)]]} \n Prediction Probability: {out_prob[temp_batch_ind+(batch_num * batch_size)]:.2f}')
                        i+=1

                if i > 19:
                    break
            
            plt.savefig('False_Positives.png')
            plt.clf()

            # False Negative
            fn_ind = [i for i, x in enumerate([a == 0 and a != b for a,b in zip(out_pred, out_true)]) if x]

            batch_num = 0
            i = 0

            for test_input, test_labels in dataloaders['deploy']:
                batch_ind = [y for y in [x - (batch_num * batch_size) for x in fn_ind] if y < 100 and y >= 0]
                if len(batch_ind) != 0:
                    plt.subplot(5,4,i+1)
                    plt.imshow(denormalize(test_input.cpu().data[batch_ind[0]], mean, std))
                    plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
                    plt.title(f'Predicted: {class_names[out_pred[batch_ind[0]+(batch_num * batch_size)]]}, Actual: {class_names[out_true[batch_ind[0]+(batch_num * batch_size)]]} \n Prediction Probability: {1-out_prob[batch_ind[0]+(batch_num * batch_size)]:.2f}')
                    i+=1

                batch_num+=1
                if i > 19:
                    break

            plt.savefig('False_Negatives.png')
            plt.clf()

    else:
        # Inference
        # Locate and Load data
        unique_ids = glob.glob(f"{image_dir}/**/*.jpg", recursive=True)

        print(f"Attempting to classify {len(unique_ids)} images.")

        # Transform by resizing, converting to grayscale and normalising
        # This is the same image processing that we perform on the training examples
        w, h = 672, 295
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Original image size and post-crop dimensions. Standardise images and crop to center before applying other transformations
        ow, oh = 2688, 1520
        cw, ch = 2688, 1180

        deploy_transformer = transforms.Compose([
        transforms.Resize((oh,ow)),
        transforms.CenterCrop((ch,cw)),
        transforms.Grayscale(num_output_channels = 3),
        transforms.Resize((h,w)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

        deploy_ds = MyDataset(unique_ids,transform=deploy_transformer,is_labeled=False)

        # Shuffle off to allow ease of matching File names to Output Labels
        # 10% batch of default
        batch_size = 250
        deploy_dl = DataLoader(deploy_ds, batch_size=batch_size,shuffle=False,collate_fn = collate_fn_inf)  
        dataloaders = {'deploy' : deploy_dl}

        dataset_sizes = {'deploy' : len(deploy_ds)}
        class_names = ['NONPILING','PILING']

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load model
        model_ft = models.resnet18()

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)

        model_ft.load_state_dict(torch.load(model_dat, weights_only = True))

        # Use the evaluation/inference mode of the model
        model_ft.eval()

        out_pred = []
        out_prob = []
        out_paths = []
        
        for inputs, paths in tqdm(dataloaders['deploy']):

            with torch.no_grad():
                inputs = inputs.to(device)
                model_ft = model_ft.to(device)

                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                probabilities = F.softmax(outputs, dim=1)[:, 1]
        
            out_pred.extend(preds.tolist())
            out_prob.extend(probabilities.tolist())
            out_paths.extend(paths)

        with open ('Classification_list.csv','w',newline='') as csvfile:
            my_writer = csv.writer(csvfile, delimiter = ',')
            my_writer.writerow(['File Path', 'Classification'])

            for i in range(len(out_pred)):
                pred_paths = out_paths[i]
                prediction = out_pred[i]

                my_writer.writerow([pred_paths, prediction])

#--------------------------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Perform Inference, either with or without validation metrics.")

    # Positional arguments (required)
    parser.add_argument("directory", type=str, help="Path to the input image directory.")
    parser.add_argument("model_dat", type=str, help="Path to, and name of, the model dat file.")

    # Optional arguments
    parser.add_argument(
        "-v", "--validate", action="store_true", help="Enable validation mode."
    )
    parser.add_argument(
        "-n",
        "--nsamples",
        type=int,
        help="Number of samples to use, per class, for validation.",
    )
    parser.add_argument(
        "-m", "--misclass", action="store_true", help="When validating, enables the generation of examples of False Positives and False Negatives for inspection."
    )
    args = parser.parse_args()
    main(args)