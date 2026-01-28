import numpy as np
import torch
import warnings
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Invertd, SaveImaged
from monai.networks.nets import SwinUNETR
from monai.data import CacheDataset, DataLoader
from pathlib import Path
from collections import defaultdict
import argparse
import pandas as pd
import os
import glob

from utils import (
    NormalizeWithMean,
    LoadDCMImagePath,
    custom_collate,
    spect_ac_recon,
    apply_post_transforms_ac,
    ensemble_across_models,
    get_ac_images_from_models,
    ensemble_image_regression
)
from evaluate import calculate_metrics_final

warnings.filterwarnings("ignore")

def reconstruct_ac(
    input_type,
    num_input,
    outpath,
    test_loader,
    prediction_dir_atm,
    val_transforms_save,
    atm_ensemble,
    osem44,
    raw_images,
    atm,
    prediction_dir_ac,
    args,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dicts = [{"image": os44, "raw": raw, "label": at, "output": atme} for os44, raw, at, atme in zip(osem44, raw_images, atm, atm_ensemble)]
    test_transforms = Compose([
        LoadImaged(keys=["image", "label", "output"]),
        LoadDCMImagePath(keys=["raw"]),
        EnsureChannelFirstd(keys=["image", "label", "output"]),
        NormalizeWithMean(keys=["image"]),
    ])
    test_ds = CacheDataset(data=data_dicts, transform=test_transforms, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=custom_collate, shuffle=False, num_workers=args.num_workers)
    for val_data in test_loader:
        val_inputs, val_raw, val_labels, atm_output = (
            val_data["image"].to(device),
            val_data["raw"],
            val_data["label"].to(device),
            val_data["output"].to(device),
        )
        if num_input == 1:
            recon_ac_batch = spect_ac_recon(
                val_raw, atm_output, 1, num_input, input_type, val_labels, args.colimator, args.energy_keV
            )
            batch = {"image": val_inputs, "label": val_labels, "pred": recon_ac_batch}
            for key in batch:
                batch[key] = batch[key][0]
            os.makedirs(prediction_dir_ac, exist_ok=True)
            apply_post_transforms_ac(batch, os.path.join(prediction_dir_ac), val_transforms_save)
        else:
            raw = val_data["raw"]
            val_labels = val_data["label"].to(device)
            recon_ac_44_batch, recon_ac_66_batch, recon_ac_88_batch = spect_ac_recon(
                raw, atm_output, 1, num_input, input_type, val_labels, 'G8-LEHR', args.energy_keV
            )
            prediction_dir_ac_44 =prediction_dir_ac[0]
            os.makedirs(prediction_dir_ac_44, exist_ok=True)
            prediction_dir_ac_66 = prediction_dir_ac[1]
            os.makedirs(prediction_dir_ac_66, exist_ok=True)
            prediction_dir_ac_88 = prediction_dir_ac[2]
            os.makedirs(prediction_dir_ac_88, exist_ok=True)
            batch = {"image": val_inputs, "label": val_labels, "pred": recon_ac_44_batch}
            for key in batch:
                batch[key] = batch[key][0]
            apply_post_transforms_ac(batch, os.path.join(prediction_dir_ac_44), val_transforms_save)
            batch = {"image": val_inputs, "label": val_labels, "pred": recon_ac_66_batch}
            for key in batch:
                batch[key] = batch[key][0]
            apply_post_transforms_ac(batch, os.path.join(prediction_dir_ac_66), val_transforms_save)
            batch = {"image": val_inputs, "label": val_labels, "pred": recon_ac_88_batch}
            for key in batch:
                batch[key] = batch[key][0]
            apply_post_transforms_ac(batch, os.path.join(prediction_dir_ac_88), val_transforms_save)


def inference_model(
    fold,
    model_name,
    input_type,
    num_input,
    outpath,
    test_loader,
    prediction_dir_atm,
    loss_function,
    val_transforms_save,
    metrics,
    args,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        img_size=(64, 64, 64),
        spatial_dims=3,
        in_channels=num_input,
        out_channels=1,
        use_v2=True,
    )
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(outpath, model_name)))
    model.eval()
    with torch.no_grad():
        for val_data in test_loader:
            if num_input == 1:
                val_inputs, val_ac, val_raw, val_labels = (
                    val_data["image"].to(device),
                    val_data["ac"].to(device),
                    val_data["raw"],
                    val_data["label"].to(device),
                )
                output = model(val_inputs)
                if args.indirect:
                    output = torch.clamp(output, min=0)
                    calculate_metrics_final(model_name, val_labels, output, metrics, 'ATM', loss_function, device)
                    batch = {"image": val_inputs, "label": val_labels, "pred": output}
                    for key in batch:
                        batch[key] = batch[key][0]
                    os.makedirs(os.path.join(prediction_dir_atm, model_name), exist_ok=True)
                    apply_post_transforms_ac(batch, os.path.join(prediction_dir_atm, model_name), val_transforms_save)
                else:
                    calculate_metrics_final(model_name, val_ac, output, metrics, input_type, loss_function, device)
                    batch = {"image": val_inputs, "label": val_labels, "pred": output}
                    for key in batch:
                        batch[key] = batch[key][0]
                    prediction_dir_ac = outpath + "/predictions/" + input_type
                    os.makedirs(prediction_dir_ac, exist_ok=True)
                    os.makedirs(os.path.join(prediction_dir_ac, model_name), exist_ok=True)
                    apply_post_transforms_ac(batch, os.path.join(prediction_dir_ac, model_name), val_transforms_save)
            else:
                if args.indirect:
                    image_44 = val_data["image_44"].to(device)
                    image_66 = val_data["image_66"].to(device)
                    image_88 = val_data["image_88"].to(device)
                    val_inputs = torch.cat([image_44, image_66, image_88], dim=1)
                    val_labels = val_data["label"].to(device)
                    output = model(val_inputs)
                    output = torch.clamp(output, min=0)
                    calculate_metrics_final(model_name, val_labels, output, metrics, 'ATM', loss_function, device)
                    batch = {"image": val_inputs, "label": val_labels, "pred": output}
                    for key in batch:
                        batch[key] = batch[key][0]
                    os.makedirs(os.path.join(prediction_dir_atm, model_name), exist_ok=True)
                    apply_post_transforms_ac(batch, os.path.join(prediction_dir_atm, model_name), val_transforms_save)


def tester(args):
    input_type=args.input_type
    num_input=args.num_input
    if args.indirect:
        if args.ac_loss and args.atm_loss:
            outpath = os.path.join(args.exp_dir,args.exp,'ATM_AC_loss',input_type)
    Path(outpath).mkdir(parents=True, exist_ok=True)  # create output directory to store model checkpoints
       # Directories
    path_test=args.data_path_test
    osem44 = sorted(glob.glob(os.path.join(path_test, "NAC/OSEM_4I4S", "*.nii.gz")))
    osem66 = sorted(glob.glob(os.path.join(path_test, "NAC/OSEM_6I6S", "*.nii.gz")))
    osem88 = sorted(glob.glob(os.path.join(path_test, "NAC/OSEM_8I8S", "*.nii.gz")))
    atm = sorted(glob.glob(os.path.join(path_test, "ATM", "*.nii.gz")))
    ac44 = sorted(glob.glob(os.path.join(path_test, "MAC/OSEM_4I4S", "*.nii.gz")))
    ac66 = sorted(glob.glob(os.path.join(path_test, "MAC/OSEM_6I6S", "*.nii.gz")))
    ac88 = sorted(glob.glob(os.path.join(path_test, "MAC/OSEM_8I8S", "*.nii.gz")))
    raw_images= sorted(glob.glob(os.path.join(path_test, "NM/", "*.*")))
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
        
    test_ds = CacheDataset(
        data=data_dicts, transform=val_transforms, num_workers=args.num_workers
    )
    test_loader = DataLoader(test_ds, batch_size=1,collate_fn=custom_collate, shuffle=False, num_workers=args.num_workers)
   
    # create the model
    # create the loss function
    loss_function = torch.nn.L1Loss()

    # Initialize best metrics dictionaries
    model_dir = args.model_dir
    model_files = [
            f for f in os.listdir(model_dir) if f.endswith(".pt")
        ]
    evaluated_metrics = defaultdict(lambda: defaultdict(list))

    fold=0
    for model_name in model_files:
        print(model_name)    
        inference_model(fold, model_name, input_type, num_input, model_dir, test_loader, prediction_dir_atm, loss_function,val_transforms_save,evaluated_metrics,args)
    if args.indirect:
        if num_input==1:
            ensemble_across_models(prediction_dir_atm, model_files )
            atm_ensemble = sorted(glob.glob(os.path.join(prediction_dir_atm, "*.nii.gz")))
            reconstruct_ac(input_type,num_input, outpath,test_loader,prediction_dir_atm,val_transforms_save,atm_ensemble, osem44,raw_images,atm,prediction_dir_ac,args)
        else:
            ensemble_across_models(prediction_dir_atm,  model_files )
            atm_ensemble = sorted(glob.glob(os.path.join(prediction_dir_atm, "*.nii.gz")))
            reconstruct_ac(input_type,num_input, outpath,test_loader,prediction_dir_atm,val_transforms_save,atm_ensemble, osem44,raw_images,atm,[prediction_dir_ac_44,prediction_dir_ac_66,prediction_dir_ac_88],args)


def __main__():
    parser = argparse.ArgumentParser()

    # data loader arguments
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of workers to use in data loader",
    )
    parser.add_argument(
        "--data_path_test",
        default='/path/to/NIfTI/folders',  # Update this path to your test data directory
        type=Path,
        help="Path to the training set",
    )
    parser.add_argument("--exp_dir", default='path/to/output',type=Path, help="output directory to save test images, same as train")
    parser.add_argument(
        "--exp",
        default='external',
        type=str,
        help="experiment name (a folder will be created with this name to store the results)",
    )
    parser.add_argument("--model_dir", default="/path/to/models_dir", type=Path, help="Path to the directory containing the trained models") 
    parser.add_argument(
        "--num_input", default=3, type=int, help="Number of inputs 1 or 3"
    )
    parser.add_argument(
        "--input_type", default='OSEM_3', type=str, help="Reconstracution name: OSEM_4I4S 'OSEM_6I6S','OSEM_8I8S','OSEM_3'"
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
        "--colimator", default='G8-LEHR', type=str, help="Colimator type"
    )
    parser.add_argument(
        "--energy_keV", default=140.5, type=float, help="Photopeak energy in keV used for PSF modelling"
    )
    #
    args = parser.parse_args()
    tester(args)
if __name__ == "__main__":
    __main__()
