import numpy as np
import torch
import warnings
from monai.utils import set_determinism
from monai.networks.nets import SwinUNETR
from pathlib import Path
import argparse
from monai.data import CacheDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import sys
from datetime import datetime
from collections import defaultdict
import glob
import time
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Invertd, SaveImaged

from utils import NormalizeWithMean, LoadDCMImagePath, custom_collate, spect_ac_recon, move_files_up
from evaluate import calculate_metrics, calculate_metrics_final, save_best_model, evaluate_model
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
warnings.filterwarnings("ignore")

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return (checkpoint['epoch'], checkpoint['best_metrics'], checkpoint['best_metric_epochs'], 
            checkpoint['epoch_loss_values'], checkpoint['val_epoch_loss_values'])

def trainer(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    outpath = os.path.join(args.exp_dir, args.exp,"ATM_AC_loss",args.input_type)
    Path(outpath).mkdir(parents=True, exist_ok=True)  # create output directory to store model checkpoints
    now = datetime.now()
    date = now.strftime("%m-%d-%y_%H-%M")
    batch_size=args.batch_size
    input_type=args.input_type
    num_input=args.num_input
    # Directories
    path_train=args.data_path_train
    osem44 = sorted(glob.glob(os.path.join(path_train, "NAC/OSEM_4I4S", "*.nii.gz")))
    osem66 = sorted(glob.glob(os.path.join(path_train, "NAC/OSEM_6I6S", "*.nii.gz")))
    osem88 = sorted(glob.glob(os.path.join(path_train, "NAC/OSEM_8I8S", "*.nii.gz")))
    atm = sorted(glob.glob(os.path.join(path_train, "ATM", "*.nii.gz")))
    ac44 = sorted(glob.glob(os.path.join(path_train, "MAC/OSEM_4I4S", "*.nii.gz")))
    ac66 = sorted(glob.glob(os.path.join(path_train, "MAC/OSEM_6I6S", "*.nii.gz")))
    ac88 = sorted(glob.glob(os.path.join(path_train, "MAC/OSEM_8I8S", "*.nii.gz")))
    raw_images= sorted(glob.glob(os.path.join(path_train, "NM/", "*.*")))
    # Data Transforms
    prediction_dir = outpath+"/predictions/"
    os.makedirs(prediction_dir, exist_ok=True)  # Ensure the directory exists
    prediction_dir_atm = outpath+"/predictions/ATM"
    os.makedirs(prediction_dir_atm, exist_ok=True)  # Ensure the directory exists

    if num_input==1:
        prediction_dir_ac = outpath+"/predictions/"+input_type
        os.makedirs(prediction_dir_ac, exist_ok=True)  # Ensure the directory exists
    else:
        prediction_dir_ac_44 = outpath+"/predictions/OSEM_4I4S"
        os.makedirs(prediction_dir_ac_44, exist_ok=True)  # Ensure the directory exists
        prediction_dir_ac_66 = outpath+"/predictions/OSEM_6I6S"
        os.makedirs(prediction_dir_ac_66, exist_ok=True)  # Ensure the directory exists
        prediction_dir_ac_88 = outpath+"/predictions/OSEM_8I8S"
        os.makedirs(prediction_dir_ac_88, exist_ok=True)  # Ensure the directory exists
    # Load data
    num_input=args.num_input
    input_type=args.input_type
    if num_input==1:
        if input_type=='OSEM_4I4S':
            data_dicts = [{"image": os44,"ac":a44,"raw":raw ,"label": at} for os44,a44,raw ,at in zip(osem44, ac44,raw_images,atm)]
        elif input_type=='OSEM_6I6S':
            data_dicts = [{"image": os66,"ac":a66 , "raw":raw ,"label": at} for os66,a66,raw, at in zip(osem66,ac66, raw_images,atm)]
        elif input_type=='OSEM_8I8S':
             data_dicts = [{"image": os88, "ac":a88 ,"raw":raw ,"label": at} for os88, a88,raw,at in zip(osem88,ac88, raw_images,atm)]
        train_transforms = Compose([
           LoadImaged(keys=["image","ac", "label"]),
           LoadDCMImagePath(keys=["raw"]),
           EnsureChannelFirstd(keys=["image","ac",  "label"]),
           NormalizeWithMean(keys=["image"])  # Add the custom normalization transform
        ])
        
        val_transforms = Compose([
           LoadImaged(keys=["image","ac", "label"]),
           LoadDCMImagePath(keys=["raw"]),
           EnsureChannelFirstd(keys=["image","ac",  "label"]),
           NormalizeWithMean(keys=["image"])  # Add the custom normalization transform
        ])
        val_transforms_save = Compose([
           LoadImaged(keys=["image", "label"]),
           EnsureChannelFirstd(keys=["image", "label"]),
           NormalizeWithMean(keys=["image"])  # Add the custom normalization transform
        ])

    else:
        data_dicts = [{"image_44": os44, "image_66": os66, "image_88": os88, "ac44": a44, "ac66": a66, "ac88": a88, "raw": raw, "label": at} for os44, os66, os88, a44, a66, a88, raw, at in zip(osem44, osem66, osem88, ac44, ac66, ac88, raw_images, atm)]    # Cross-validation
        train_transforms = Compose([
           LoadImaged(keys=["image_44","image_66", "image_88", "ac44", "ac66", "ac88","label"]),
           LoadDCMImagePath(keys=["raw"]),
           EnsureChannelFirstd(keys=["image_44","image_66", "image_88","ac44", "ac66", "ac88","label"]),
           NormalizeWithMean(keys=["image_44","image_66", "image_88"])  # Add the custom normalization transform
        ])
        val_transforms = Compose([
           LoadImaged(keys=["image_44","image_66", "image_88","ac44", "ac66", "ac88","label"]),
           LoadDCMImagePath(keys=["raw"]),
           EnsureChannelFirstd(keys=["image_44","image_66", "image_88","ac44", "ac66", "ac88","label"]),
           NormalizeWithMean(keys=["image_44","image_66", "image_88"])  # Add the custom normalization transform
        ])
        val_transforms_save = Compose([
           LoadImaged(keys=["image_44","image_66", "image_88","label"]),
           EnsureChannelFirstd(keys=["image_44","image_66", "image_88","label"]),
           NormalizeWithMean(keys=["image_44","image_66", "image_88"])  # Add the custom normalization transform
        ])
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = defaultdict(lambda: defaultdict(list))
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_dicts)):
        if fold < args.start_fold:
            continue  # Skip folds before the start_fold
        print(f"Fold {fold+1}/{n_splits}")
    
        train_data0 =[data_dicts[i] for i in train_idx]
        test_data = [data_dicts[i] for i in test_idx]
        train_data, val_data = train_test_split(train_data0, test_size = 0.10, random_state=42,shuffle=True)

        train_ds = CacheDataset(
            data=train_data, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,collate_fn=custom_collate, shuffle=True, num_workers=args.num_workers)
    
        val_ds = CacheDataset(
            data=val_data, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
        )
        val_loader = DataLoader(val_ds, batch_size=1,collate_fn=custom_collate, shuffle=True, num_workers=args.num_workers)
        test_ds = CacheDataset(
            data=test_data, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
        )
        test_loader = DataLoader(test_ds, batch_size=1,collate_fn=custom_collate, shuffle=True, num_workers=args.num_workers)
   
        # create the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_input=args.num_input
        model = SwinUNETR(img_size=(64,64,64),
            spatial_dims=3,
            in_channels=args.num_input,
            out_channels=1,
            use_v2=True
        )
        model.to(device)

        print("#model_params:", np.sum([len(p.flatten()) for p in model.parameters()]))
        # create the loss function
        loss_function = torch.nn.L1Loss()
    
        # create the optimizer and the learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    
        # start a typical PyTorch training loop
        val_interval = 2  # doing validation every 2 epochs
        tic = time.time()
        epoch_loss_values=[]
        val_epoch_loss_values=[]

        # Initialize best metrics dictionaries
        best_metrics = {
            "ATM": {"ssim": -float("inf"), "mae": float("inf"), "mse": float("inf"), "psnr": -float("inf"), "rae": float("inf")},
            "OSEM_4I4S": {"ssim": -float("inf"), "mae": float("inf"), "mse": float("inf"), "psnr": -float("inf"), "rae": float("inf")},
            "OSEM_6I6S": {"ssim": -float("inf"), "mae": float("inf"), "mse": float("inf"), "psnr": -float("inf"), "rae": float("inf")},
            "OSEM_8I8S": {"ssim": -float("inf"), "mae": float("inf"), "mse": float("inf"), "psnr": -float("inf"), "rae": float("inf")},
        }
        
        best_metric_epochs = {
            "ATM": {"ssim": -float("inf"), "mae": float("inf"), "mse": float("inf"), "psnr": -float("inf"), "rae": float("inf")},
            "OSEM_4I4S": {"ssim": -float("inf"), "mae": float("inf"), "mse": float("inf"), "psnr": -float("inf"), "rae": float("inf")},
            "OSEM_6I6S": {"ssim": -float("inf"), "mae": float("inf"), "mse": float("inf"), "psnr": -float("inf"), "rae": float("inf")},
            "OSEM_8I8S": {"ssim": -float("inf"), "mae": float("inf"), "mse": float("inf"), "psnr": -float("inf"), "rae": float("inf")},
        }
        start_epoch = 0
        if args.resume_checkpoint:
            checkpoint_path = os.path.join(outpath, f"SwinUNETR_Fold{fold}_checkpoint.pth.tar")
            if os.path.isfile(checkpoint_path):
                start_epoch, best_metrics, best_metric_epochs, epoch_loss_values, val_epoch_loss_values = load_checkpoint(
                    torch.load(checkpoint_path), model, optimizer
                )
                print(f"Resumed training from checkpoint at epoch {start_epoch}...")
        for epoch in range(start_epoch, args.num_epochs):         
            model.train()
            epoch_loss = 0
            step = 0
            writer = SummaryWriter(
                f"{outpath}/fold_{fold}"
            )  # create a date directory within the output directory for storing training logs

            train_bar = tqdm(train_loader)  # Progress Bar
            for batch_data in train_bar:
                    step += 1
                    if num_input==1:
                        inputs, ac,raw,labels = (
                            batch_data["image"].to(device),
                            batch_data["ac"].to(device),  
                            batch_data["raw"],                           
                            batch_data["label"].to(device),
                        )
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = 0
                        if args.indirect:
                            if args.ac_loss:
                                recon_ac_batch=spect_ac_recon(raw, outputs,batch_size,num_input,input_type,labels, args.colimator)
                                loss_ac = loss_function(recon_ac_batch, ac)
                                loss += loss_ac
                            if args.atm_loss:
                                loss_atm = loss_function(outputs, labels)
                                loss += loss_atm
                        else:
                            print('Train in Direct method:')
                            loss_ac = loss_function(outputs, ac)
                            loss += loss_ac
                    else:
                        image_44 = batch_data["image_44"].to(device)  # shape: 4*1*64*64*64
                        image_66 = batch_data["image_66"].to(device)  # shape: 4*1*64*64*64
                        image_88 = batch_data["image_88"].to(device)  # shape: 4*1*64*64*64
                        ac_44 = batch_data["ac44"].to(device)  # shape: 4*1*64*64*64
                        ac_66 = batch_data["ac66"].to(device)  # shape: 4*1*64*64*64
                        ac_88 = batch_data["ac88"].to(device)  # shape: 4*1*64*64*64
                        inputs = torch.cat([image_44, image_66, image_88], dim=1)
                        labels = batch_data["label"].to(device)
                        raw=batch_data["raw"]
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = 0
                        if args.indirect:
                            if args.ac_loss:
                                recon_ac_44_batch, recon_ac_66_batch, recon_ac_88_batch=spect_ac_recon(raw, outputs,batch_size,num_input,input_type,labels, args.colimator)
                                loss_44 = loss_function(recon_ac_44_batch, ac_44)
                                loss_66 = loss_function(recon_ac_66_batch, ac_66)
                                loss_88 = loss_function(recon_ac_88_batch, ac_88)
                                loss += loss_44+loss_66+loss_88
                            if args.atm_loss:
                                loss_atm = loss_function(outputs, labels)
                                loss += loss_atm
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    train_bar.set_description(desc=f"Epoch {epoch + 1}/{args.num_epochs}, Train_loss: {loss.item():.4f}")  # progress bar description                    #sys.stdout.flush()
            scheduler.step()
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            writer.add_scalar("train_loss", epoch_loss, epoch + 1)
            # validation
            if num_input==1: 
                metrics = {
                    "ATM_ssim": [], "ATM_mae": [], "ATM_mse": [], "ATM_psnr": [], "ATM_rae": [], "ATM_nmse": [], "ATM_nmae": [],
                    f"{input_type}_ssim": [], f"{input_type}_mae": [], f"{input_type}_mse": [], f"{input_type}_psnr": [], f"{input_type}_rae": [], f"{input_type}_nmse": [], f"{input_type}_nmae": [],
                }
            else:
                metrics = {
                    "ATM_ssim": [], "ATM_mae": [], "ATM_mse": [], "ATM_psnr": [], "ATM_rae": [], "ATM_nmse": [], "ATM_nmae": [],
                    "OSEM_4I4S_ssim": [], "OSEM_4I4S_mae": [], "OSEM_4I4S_mse": [], "OSEM_4I4S_psnr": [], "OSEM_4I4S_rae": [], "OSEM_4I4S_nmse": [], "OSEM_4I4S_nmae": [],
                    "OSEM_6I6S_ssim": [], "OSEM_6I6S_mae": [], "OSEM_6I6S_mse": [], "OSEM_6I6S_psnr": [], "OSEM_6I6S_rae": [], "OSEM_6I6S_nmse": [], "OSEM_6I6S_nmae": [],
                    "OSEM_8I8S_ssim": [], "OSEM_8I8S_mae": [], "OSEM_8I8S_mse": [], "OSEM_8I8S_psnr": [], "OSEM_8I8S_rae": [], "OSEM_8I8S_nmse": [], "OSEM_8I8S_nmae": [],
                }
            if (epoch + 1) % val_interval == 0:
                model.eval()
                val_epoch_loss = 0
                val_step = 0
                with torch.no_grad():                   
                        for val_data in val_loader:
                            val_step += 1
                            if num_input==1:
                                val_inputs,val_ac,val_raw ,val_labels = (
                                        val_data["image"].to(device),
                                        val_data["ac"].to(device),  
                                        val_data["raw"],                           
                                        val_data["label"].to(device),
                                        )
                                output = model(val_inputs)
                                vloss = 0
                                if args.indirect:
                                    if args.ac_loss:
                                        recon_ac_batch=spect_ac_recon(val_raw, output,1,num_input,input_type,val_labels, args.colimator)
                                        vloss += loss_function(recon_ac_batch, val_ac)
                                        calculate_metrics(val_ac, recon_ac_batch, metrics, input_type,loss_function,device)
                                    if args.atm_loss:
                                        vloss_atm = loss_function(output, val_labels)
                                        vloss+=vloss_atm
                                else:
                                    print('Validate in Direct method:')
                                    vloss_ac = loss_function(output, val_ac)
                                    calculate_metrics(val_ac, output, metrics, input_type,loss_function,device)
                                    vloss += vloss_ac

                            else:
                                image_44 = val_data["image_44"].to(device)  # shape: 4*1*64*64*64
                                image_66 = val_data["image_66"].to(device)  # shape: 4*1*64*64*64
                                image_88 = val_data["image_88"].to(device)  # shape: 4*1*64*64*64
                                val_inputs = torch.cat([image_44, image_66, image_88], dim=1)
                                ac_44 = val_data["ac44"].to(device)  # shape: 4*1*64*64*64
                                ac_66 = val_data["ac66"].to(device)  # shape: 4*1*64*64*64
                                ac_88 = val_data["ac88"].to(device)  # shape: 4*1*64*64*64
                                raw=batch_data["raw"]
    
                                val_labels = val_data["label"].to(device)
                                output = model(val_inputs)
                                vloss = 0
                                if args.indirect:
                                    if args.ac_loss:
                                        recon_ac_44_batch, recon_ac_66_batch, recon_ac_88_batch=spect_ac_recon(raw, output,1,num_input,input_type,val_labels,args.colimator)
                                        calculate_metrics(ac_44, recon_ac_44_batch, metrics, 'OSEM_4I4S',loss_function,device)
                                        calculate_metrics(ac_66, recon_ac_66_batch, metrics, 'OSEM_6I6S',loss_function,device)
                                        calculate_metrics(ac_88, recon_ac_88_batch, metrics, 'OSEM_8I8S',loss_function,device)
         
                                        loss_44 = loss_function(recon_ac_44_batch, ac_44)
                                        loss_66 = loss_function(recon_ac_66_batch, ac_66)
                                        loss_88 = loss_function(recon_ac_88_batch, ac_88)
                                        vloss+=loss_44+loss_66+loss_88
                                    if args.atm_loss:
                                        vloss_atm = loss_function(output, val_labels)
                                        vloss+=vloss_atm
                            val_epoch_loss += vloss.item()
                            output=torch.clamp(output, min=0)
                            calculate_metrics(val_labels, output, metrics, 'ATM',loss_function,device)
                        val_epoch_loss /= val_step
                        val_epoch_loss_values.append(val_epoch_loss)
                        if num_input == 1:
                            if args.indirect:
                                for metric in [ "mae"]:
                                    save_best_model(metric, metrics, "ATM",best_metrics,best_metric_epochs,epoch,model,outpath,fold)
                                    if args.ac_loss:
                                        save_best_model(metric, metrics, input_type,best_metrics,best_metric_epochs,epoch,model,outpath,fold)
                            else:
                                for metric in [ "mae"]:
                                    save_best_model(metric, metrics, input_type,best_metrics,best_metric_epochs,epoch,model,outpath,fold)                
                        else:
                            if args.indirect:
                                for metric in [ "mae"]:
                                    save_best_model(metric, metrics, "ATM",best_metrics,best_metric_epochs,epoch,model,outpath,fold)
    
            # Save checkpoint after every epoch
            checkpoint_path = os.path.join(outpath, f"SwinUNETR_Fold{fold}_checkpoint.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metrics': best_metrics,
                'best_metric_epochs': best_metric_epochs,
                'epoch_loss_values': epoch_loss_values,
                'val_epoch_loss_values': val_epoch_loss_values,
            }, checkpoint_path)
        # Save the final model state
        #torch.save(model.state_dict(), os.path.join(outpath, f"SwinUNETR_Fold{fold}_lastepock.pt"))

        toc=time.time()
        diff_time=toc-tic
        print(f'Trianing time: {diff_time}')
        model_files = [
                f for f in os.listdir(outpath) if f.startswith(f"SwinUNETR_Fold{fold}") and f.endswith(".pt")
            ]
        evaluated_metrics = defaultdict(lambda: defaultdict(list))
        for model_name in model_files:
            print(model_name)    
            evaluated_metrics=evaluate_model(fold, model_name, input_type, num_input, outpath, test_loader, prediction_dir_atm, loss_function,val_transforms_save,evaluated_metrics,args)
        x_val = np.linspace(0, len(epoch_loss_values), len(val_epoch_loss_values)+1)
        plt.figure("train", (12, 6))
        plt.title("Epoch Average Loss")
        plt.plot(epoch_loss_values,'r',linewidth=3.0, label='Training loss')
        plt.plot(x_val[1:], val_epoch_loss_values, 'b', linewidth=3.0, label='Validation loss')
        plt.legend(fontsize=16)
        plt.xlabel('Epochs ',fontsize=16)    
        plt.ylabel('Loss',fontsize=16)
        plt.savefig(f'{outpath}/fold_{fold}/fold_{fold}_loss.jpg',
                dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format='jpg', bbox_inches=None, pad_inches=0.1)
        plt.close() 
        # Inside the loop where you evaluate and collect metrics
        for metric, models in evaluated_metrics.items():
            for model, values in models.items():
                fold_metrics[metric][model].append(np.mean(values))
        metrics_data = {metric: values for metric, values in fold_metrics.items()}

        # Create a pandas DataFrame
        df = pd.DataFrame(metrics_data)
        df.to_excel(f'{outpath}/folds{fold}_metrics.xlsx')
        best_metrics_df=pd.DataFrame(best_metrics)
        best_metrics_df.to_excel(f'{outpath}/folds{fold}_best_metrics_df.xlsx')
        best_metric_epochs_df=pd.DataFrame(best_metric_epochs)
        best_metric_epochs_df.to_excel(f'{outpath}/folds{fold}_best_metric_epochs_df.xlsx')
    for folder in ["ATM", "OSEM_4I4S", "OSEM_6I6S", "OSEM_8I8S"]:
        full_path = os.path.join(prediction_dir, folder)
        if os.path.isdir(full_path):
            print(f"Processing {full_path}...")
            move_files_up(full_path, n_splits)
        else:
            print(f"Skipping {folder} — not found.")

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Data loader batch size (batch_size>1 is suitable for varying input size",
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of workers to use in data loader",
    )
    parser.add_argument(
        "--cache_rate",
        default=0.4,
        type=float,
        help="The fraction of the data to be cached when being loaded",
    )
    parser.add_argument(
        "--data_path_train",
        default='/path/to/NIfTI/train/folder',
        type=Path,
        help="Path to the training set",
    )
    parser.add_argument("--num_epochs", default=300,type=int, help="number of training epochs")
    parser.add_argument("--exp_dir", default='/path/to/output',type=Path, help="output directory to save training logs")
    parser.add_argument(
        "--exp",
        default='train',
        type=str,
        help="experiment name (a folder will be created with this name to store the results)",
    )
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--lr_step_size", default=40, type=int, help="decay learning rate every lr_step_size epochs")
    parser.add_argument(
        "--lr_gamma",
        default=0.1,
        type=float,
        help="every lr_step_size epochs, decay learning rate by a factor of lr_gamma",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="ridge regularization factor")
    parser.add_argument(
        "--resume_checkpoint", default=True , type=bool, help="if True, training statrts from a model checkpoint"
    )
    parser.add_argument(
        "--num_input", default=3, type=int, help="Number of inputs 1 or 3"
    )
    parser.add_argument(
        "--input_type", default='OSEM_3', type=str, help="Reconstracution name"
    )
    parser.add_argument(
    "--start_fold", default=0, type=int, help="Strat fold number"
    )
    parser.add_argument(
        "--atm_loss", default=True, type=bool, help="if True, add atm loss"
    )
    parser.add_argument(
        "--ac_loss", default=True, type=bool, help="if True, add atm loss"
    )
    parser.add_argument(
        "--indirect", default=True, type=bool, help="if True, training in inderect otherwise in direct method"
    )
    parser.add_argument(
        "--colimator", default='SY-LEHR', type=str, help="Colimator type"
    )

    args = parser.parse_args()
    trainer(args)
if __name__ == "__main__":
    __main__()


