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
import shutil
import nibabel as nib

from utils import (
    NormalizeWithMean,
    LoadDCMImagePath,
    custom_collate,
    spect_ac_recon,
    apply_post_transforms_ac,
    get_ac_images_from_models,
    ensemble_image_regression,
    ensemble_across_models,
    convert_dicom_to_nifti,
    reconstruct_nm_to_nac
)
from evaluate import calculate_metrics_final

warnings.filterwarnings("ignore")
def save_dcm(
    save_path: str,
    object: torch.Tensor,
    file_NM: str,
    recon_name: str = '',
    return_ds: bool = False,
    single_dicom_file: bool = False,
    scale_by_number_projections: bool = False
    ) -> None:
    """Saves the reconstructed object `object` to a series of DICOM files in the folder given by `save_path`. Requires the filepath of the projection data `file_NM` to get Study information.

    Args:
        object (torch.Tensor): Reconstructed object of shape [Lx,Ly,Lz].
        save_path (str): Location of folder where to save the DICOM output files.
        file_NM (str): File path of the projection data corresponding to the reconstruction.
        recon_name (str): Type of reconstruction performed. Obtained from the `recon_method_str` attribute of a reconstruction algorithm class.
        return_ds (bool): If true, returns the DICOM dataset objects instead of saving to file. Defaults to False.
    """
    import pydicom
    from pydicom.uid import generate_uid
    from pytomography.io.shared import (
        create_ds
    )
    from pytomography.io.SPECT.dicom import get_metadata
    import copy

    if not return_ds:
        try:
            Path(save_path).resolve().mkdir(parents=True, exist_ok=False)
        except:
            raise Exception(
                f"Folder {save_path} already exists; new folder name is required."
            )
    # Convert tensor image to numpy array
    ds_NM = pydicom.dcmread(file_NM)
    SOP_instance_UID = generate_uid()
    if single_dicom_file:
        SOP_class_UID = '1.2.840.10008.5.1.4.1.1.20'
        modality = 'NM'
        imagetype = "['ORIGINAL', 'PRIMARY', 'RECON TOMO', 'EMISSION']"
    else:
        SOP_class_UID = "1.2.840.10008.5.1.4.1.1.128"  # SPECT storage
        modality = 'PT'
        imagetype = None
    ds = create_ds(ds_NM, SOP_instance_UID, SOP_class_UID, modality, imagetype)
    pixel_data = torch.permute(object,(2,1,0)).cpu().numpy()
    if scale_by_number_projections:
        scale_factor = get_metadata(file_NM)[1].num_projections
        ds.RescaleSlope = 1
    else:
        scale_factor = (2**16 - 1) / pixel_data.max()
        ds.RescaleSlope = 1/scale_factor
    pixel_data *= scale_factor #maximum dynamic range
    pixel_data = pixel_data.round().astype(np.uint16)
    # Affine
    Sx, Sy, Sz = ds_NM.DetectorInformationSequence[0].ImagePositionPatient
    dx = dy = ds_NM.PixelSpacing[0]
    dz = ds_NM.PixelSpacing[1]
    if Sy == 0:
        Sx -= ds_NM.Rows / 2 * dx
        Sy -= ds_NM.Rows / 2 * dy
        # Y-Origin point at tableheight=0
        Sy -= ds_NM.RotationInformationSequence[0].TableHeight
    # Sz now refers to location of lowest slice
    Sz -= (pixel_data.shape[0] - 1) * dz
    ds.Rows, ds.Columns = pixel_data.shape[1:]
    ds.SeriesNumber = 1
    if single_dicom_file:
        ds.NumberOfFrames = pixel_data.shape[0]
    else:
        ds.NumberOfSlices = pixel_data.shape[0]
    ds.PixelSpacing = [dx, dy]
    ds.SliceThickness = dz
    ds.SpacingBetweenSlices = dz
    ds.ImageOrientationPatient = [1,0,0,0,1,0]
    # Set other things
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.ReconstructionMethod = recon_name
    if single_dicom_file:
        ds.InstanceNumber = 1
        ds.ImagePositionPatient = [Sx, Sy, Sz]
        ds.PixelData = pixel_data.tobytes()
    # Add all study data/time information if available
    for attr in ['StudyDate', 'StudyTime', 'SeriesDate', 'SeriesTime', 'AcquisitionDate', 'AcquisitionTime', 'ContentDate', 'ContentTime', 'PatientSex', 'PatientAge', 'SeriesDescription', 'Manufacturer', 'PatientWeight', 'PatientHeight']:
        if hasattr(ds_NM, attr):
            ds[attr] = ds_NM[attr]
    # Create all slices
    if not single_dicom_file:
        dss = []
        for i in range(pixel_data.shape[0]):
            # Load existing DICOM file
            ds_i = copy.deepcopy(ds)
            ds_i.InstanceNumber = i + 1
            ds_i.ImagePositionPatient = [Sx, Sy, Sz + i * dz]
            # Create SOP Instance UID unique to slice
            ds_i.SOPInstanceUID = f"{ds.SOPInstanceUID[:-3]}{i+1:03d}"
            ds_i.file_meta.MediaStorageSOPInstanceUID = ds_i.SOPInstanceUID
            # Set the pixel data
            ds_i.PixelData = pixel_data[i].tobytes()
            dss.append(ds_i)      
    if return_ds:
        if single_dicom_file:
            return ds
        else:
            return dss
    else:
        if single_dicom_file:
            # If single dicom file, will overwrite any file that is there
            ds.save_as(os.path.join(save_path, f'{ds.SOPInstanceUID}.dcm'))
        else:
            for ds_i in dss:
                ds_i.save_as(os.path.join(save_path, f'{ds_i.SOPInstanceUID}.dcm'))

def generate_atm_predictions_ensemble(
    model_dir,
    nac_dir,
    output_dir,
    num_workers
):
    """
    Generate ATM predictions using ensemble of trained models from NAC images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get all model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    print(f"Found {len(model_files)} models: {model_files}")
    
    # Multi-input case only
    nac_44_files = sorted(glob.glob(os.path.join(nac_dir, "NAC/OSEM_4I4S", "*.nii.gz")))
    nac_66_files = sorted(glob.glob(os.path.join(nac_dir, "NAC/OSEM_6I6S", "*.nii.gz")))
    nac_88_files = sorted(glob.glob(os.path.join(nac_dir, "NAC/OSEM_8I8S", "*.nii.gz")))
    
    data_dicts = [{"image_44": nac44, "image_66": nac66, "image_88": nac88} 
                 for nac44, nac66, nac88 in zip(nac_44_files, nac_66_files, nac_88_files)]
    
    val_transforms = Compose([
        LoadImaged(keys=["image_44", "image_66", "image_88"]),
        EnsureChannelFirstd(keys=["image_44", "image_66", "image_88"]),
        NormalizeWithMean(keys=["image_44", "image_66", "image_88"])
    ])
    
    # Create dataset and dataloader
    test_ds = CacheDataset(data=data_dicts, transform=val_transforms, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=custom_collate, shuffle=False, num_workers=num_workers)
    
    # Create output directory for ATM predictions
    atm_output_dir = os.path.join(output_dir, "ATM")
    os.makedirs(atm_output_dir, exist_ok=True)
    
    # Generate ATM predictions for each model
    print(f"Generating ATM predictions for {len(data_dicts)} images using {len(model_files)} models...")
    
    for model_name in model_files:
        print(f"Processing model: {model_name}")
        
        # Load the model
        model = SwinUNETR(
            img_size=(64, 64, 64),
            spatial_dims=3,
            in_channels=3,  # Multi-input only
            out_channels=1,
            use_v2=True,
        )
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
        model.eval()
        
        # Create model-specific output directory
        model_output_dir = os.path.join(atm_output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        with torch.no_grad():
            for i, val_data in enumerate(test_loader):
                # Multi-input case
                image_44 = val_data["image_44"].to(device)
                image_66 = val_data["image_66"].to(device)
                image_88 = val_data["image_88"].to(device)
                val_inputs = torch.cat([image_44, image_66, image_88], dim=1)
                
                output = model(val_inputs)
                output = torch.clamp(output, min=0)
                
                # Save ATM prediction
                batch = {"image": val_inputs, "pred": output}
                for key in batch:
                    batch[key] = batch[key][0]
                
                apply_post_transforms_ac(batch, model_output_dir, val_transforms)
                
                print(f"  Generated ATM prediction {i+1}/{len(data_dicts)} for {model_name}")
    
    print(f"ATM predictions saved to: {atm_output_dir}")
    return atm_output_dir, model_files


def generate_ac_reconstructions(
    test_loader,
    val_transforms_save,
    prediction_dir_ac,
    args
):
    """
    Generate AC reconstructions using the ensemble ATM predictions and raw data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Generating AC reconstructions for {len(test_loader)} samples...")
    
    with torch.no_grad():
        for i, val_data in enumerate(test_loader):
            # Multi-input case only
            image_44 = val_data["image_44"].to(device)
            image_66 = val_data["image_66"].to(device)
            image_88 = val_data["image_88"].to(device)
            val_inputs = torch.cat([image_44, image_66, image_88], dim=1)
            val_labels = val_data["label"].to(device)
            val_raw = val_data["raw"]
            
            # Generate AC reconstructions for all 3 settings
            recon_ac_44_batch, recon_ac_66_batch, recon_ac_88_batch = spect_ac_recon(
                val_raw, val_labels, 1, 3, 'OSEM_3', val_labels, args.colimator, args.energy_keV
            )
            # Save AC reconstructions
            settings = [
                (recon_ac_44_batch, "OSEM_4I4S"),
                (recon_ac_66_batch, "OSEM_6I6S"),
                (recon_ac_88_batch, "OSEM_8I8S")
            ]
            
            for recon_batch, setting_name in settings:
                batch_ac = {"image": val_inputs, "label": val_labels, "pred": recon_batch}
                for key in batch_ac:
                    batch_ac[key] = batch_ac[key][0]
                prediction_dir_ac_setting = os.path.join(prediction_dir_ac, setting_name)
                os.makedirs(prediction_dir_ac_setting, exist_ok=True)
                apply_post_transforms_ac(batch_ac, prediction_dir_ac_setting, val_transforms_save)

            print(f"Generated AC reconstruction {i+1}/{len(test_loader)}")
    
    print(f"AC reconstructions saved to: {prediction_dir_ac}")

def process_nm_files_test(args):
    """
    Main function to process NM raw files and generate all outputs
    """
    # Create output directory structure
    output_base = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_base, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(output_base, "NM"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "NAC"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "ATM"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "MAC"), exist_ok=True)
    
    # Copy NM raw files to output directory
    nm_files = sorted(glob.glob(os.path.join(args.nm_raw_dir, "*.*")))
    for nm_file in nm_files:
        shutil.copy2(nm_file, os.path.join(output_base, "NM"))
    
    print(f"Processing {len(nm_files)} NM raw files...")
    
    # Step 1: Reconstruct NM to NAC for each raw file
    for nm_file in nm_files:
        print(f"\nProcessing: {os.path.basename(nm_file)}")
        reconstruct_nm_to_nac(nm_file, output_base, args.colimator, args.energy_keV)
    
    # Step 2: Generating ATM predictions for each model
    print(f"\nStep 2: Generating ATM predictions for each model...")
    atm_predictions_dir, model_files = generate_atm_predictions_ensemble(
        args.model_dir,
        output_base,
        output_base,
        args.num_workers
    )
    
    # Step 3: Create ensemble ATM predictions
    print(f"\nStep 3: Creating ensemble ATM predictions...")
    ensemble_across_models(atm_predictions_dir, model_files)
    atm_ensemble = sorted(glob.glob(os.path.join(atm_predictions_dir, "*.nii.gz")))
    
    # Step 4: Prepare data for AC reconstruction
    print(f"\nStep 4: Preparing data for AC reconstruction...")
    
    # Get the ensemble ATM predictions and raw files
    raw_files = sorted(glob.glob(os.path.join(output_base, "NM", "*.*")))
    nac_44_files = sorted(glob.glob(os.path.join(output_base, "NAC/OSEM_4I4S", "*.nii.gz")))
    nac_66_files = sorted(glob.glob(os.path.join(output_base, "NAC/OSEM_6I6S", "*.nii.gz")))
    nac_88_files = sorted(glob.glob(os.path.join(output_base, "NAC/OSEM_8I8S", "*.nii.gz")))
    
    # Multi-input case only
    data_dicts = [{"image_44": nac44, "image_66": nac66, "image_88": nac88, 
                  "label": atm, "raw": raw} 
                 for nac44, nac66, nac88, atm, raw in zip(nac_44_files, nac_66_files, nac_88_files, atm_ensemble, raw_files)]
    
    val_transforms = Compose([
        LoadImaged(keys=["image_44", "image_66", "image_88", "label"]),
        LoadDCMImagePath(keys=["raw"]),
        EnsureChannelFirstd(keys=["image_44", "image_66", "image_88", "label"]),
        NormalizeWithMean(keys=["image_44", "image_66", "image_88"])
    ])
    
    val_transforms_save = Compose([
        LoadImaged(keys=["image_44", "image_66", "image_88", "label"]),
        EnsureChannelFirstd(keys=["image_44", "image_66", "image_88", "label"]),
        NormalizeWithMean(keys=["image_44", "image_66", "image_88"])
    ])
    
    # Create dataset and dataloader
    test_ds = CacheDataset(data=data_dicts, transform=val_transforms, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=custom_collate, shuffle=False, num_workers=args.num_workers)
    
    # Step 5: Generate AC reconstructions
    print(f"\nStep 5: Generating AC reconstructions...")
    
    # Create prediction directories
    prediction_dir_ac = os.path.join(output_base, "MAC")
    
    # Run AC reconstruction
    generate_ac_reconstructions(
        test_loader,
        val_transforms_save,
        prediction_dir_ac,
        args
    )
    
    print(f"\nProcessing complete!")
    print(f"Results saved in: {output_base}")
    print(f"- NM raw files: {os.path.join(output_base, 'NM')}")
    print(f"- NAC reconstructions: {os.path.join(output_base, 'NAC')}")
    print(f"- ATM predictions: {atm_predictions_dir}")
    print(f"- MAC reconstructions: {prediction_dir_ac}")

def main():
    parser = argparse.ArgumentParser(description="Inference test for NM raw files")
    
    # Input/output arguments
    parser.add_argument(
        "--nm_raw_dir",
        required=True,
        type=str,
        help="Directory containing NM raw files"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Output directory for all results"
    )
    parser.add_argument(
        "--exp_name",
        default="test_inference",
        type=str,
        help="Experiment name for output folder"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        help="Directory containing trained model files (.pt files)"
    )
    
    # Processing parameters
    parser.add_argument(
        "--colimator",
        default='G8-LEHR',
        type=str,
        help="Collimator type for SPECT reconstruction"
    )
    parser.add_argument(
        "--energy_keV",
        default=140.5,
        type=float,
        help="Photopeak energy in keV used for PSF modelling"
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers for data loading"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.nm_raw_dir):
        raise ValueError(f"NM raw directory does not exist: {args.nm_raw_dir}")
    
    if not os.path.exists(args.model_dir):
        raise ValueError(f"Model directory does not exist: {args.model_dir}")
    
    # Check if there are model files in the directory
    model_files = [f for f in os.listdir(args.model_dir) if f.endswith(".pt")]
    if not model_files:
        raise ValueError(f"No .pt model files found in directory: {args.model_dir}")
    print(f"Found {len(model_files)} model files: {model_files}")
    
    # Run the test
    process_nm_files_test(args)

if __name__ == "__main__":
    main()
