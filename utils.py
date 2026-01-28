import os
import glob
import numpy as np
import torch
import shutil
from monai.transforms import MapTransform, Transform, Compose, Invertd, SaveImaged
from collections import defaultdict


class NormalizeWithMean(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            non_zero = img[img != 0].flatten()
            if non_zero.size:
                d[key] = img / non_zero.mean()
        return d


class LoadDCMImagePath(Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not os.path.isfile(d[key]):
                raise RuntimeError(f"File {d[key]} not found.")
        return d

def absolute_percent_error_metric(predicted_image, reference_image, lower_treshold="none", upper_treshold="none"):
    if isinstance(predicted_image, list):
        predicted_image = predicted_image[0]
    if isinstance(reference_image, list):
        reference_image = reference_image[0]
    if torch.is_tensor(reference_image):
        reference_image = reference_image.cpu().detach().numpy()
    if torch.is_tensor(predicted_image):
        predicted_image = predicted_image.cpu().detach().numpy()
    if lower_treshold != "none":
        reference_image[reference_image < lower_treshold] = reference_image.min()
        predicted_image[predicted_image < lower_treshold] = predicted_image.min()
    if upper_treshold != "none":
        reference_image[reference_image > upper_treshold] = reference_image.max()
        predicted_image[predicted_image > upper_treshold] = predicted_image.max()
    with np.errstate(divide="ignore", invalid="ignore"):
        bias_map = predicted_image - reference_image
        re_percent_map = (bias_map / reference_image) * 100
        rae_percent_map = abs(bias_map / reference_image) * 100
        re_percent = np.mean(np.ma.masked_invalid(re_percent_map))
        rae_percent = np.mean(np.ma.masked_invalid(rae_percent_map))
    return rae_percent

def custom_collate(batch):
    if isinstance(batch[0], dict):
        return {key: custom_collate([d[key] for d in batch]) for key in batch[0]}
    return torch.utils.data._utils.collate.default_collate(batch)


def nmse(sr_image, gt_image):
    mse_ = (sr_image - gt_image).pow(2).mean().item()
    return mse_ / (gt_image.pow(2).max().item())


def nmae(sr_image, gt_image):
    mae_ = (sr_image - gt_image).abs().mean().item()
    return mae_ / (gt_image.abs().max().item())

def spect_ac_recon(raw,outputs,bs,num_input,input_type,labels,colimator,energy_keV=140.5):
    from pytomography.io.SPECT import dicom
    from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
    from pytomography.algorithms import OSEM
    from pytomography.projectors.SPECT import SPECTSystemMatrix
    from pytomography.likelihoods import PoissonLogLikelihood

    outputs=torch.clamp(outputs, min=0)
    recon_ac_batch = []
    recon_ac_44_batch=[]
    recon_ac_66_batch=[]
    recon_ac_88_batch=[]
    if len(raw)<bs:
        bs=len(raw)
    for i in range(bs):
        raw0 = raw[i]
        outputs0 = outputs[i]
        label0=labels[i]
        object_meta, proj_meta = dicom.get_metadata(raw0, index_peak=0)
        photopeak = dicom.get_projections(raw0, index_peak=0)
        psf_meta = dicom.get_psfmeta_from_scanner_params(colimator, energy_keV=energy_keV)
        psf_transform = SPECTPSFTransform(psf_meta)
        att_transform = SPECTAttenuationTransform(outputs0[0])
        system_matrix_ac = SPECTSystemMatrix(
                obj2obj_transforms = [att_transform,psf_transform],
                proj2proj_transforms = [],
                object_meta = object_meta,
                proj_meta = proj_meta)
        likelihood = PoissonLogLikelihood(system_matrix_ac, photopeak)
        reconstruction_algorithm = OSEM(likelihood)   
        mask_atm = np.any(label0 != 0, axis=(1, 2))[0]
        if num_input ==1:               
            if input_type == 'OSEM_4I4S':
                it = 4
            elif input_type == 'OSEM_6I6S':
                it = 6
            elif input_type == 'OSEM_8I8S':
                it = 8
            else:
                raise ValueError("Invalid input_type. Must be 'OSEM_4I4S', 'OSEM_6I6S', or 'OSEM_8I8S'.")
            recon_ac = reconstruction_algorithm(n_iters=it, n_subsets=it)
            recon_ac[:, :, ~mask_atm] = 0

            recon_ac_batch.append(recon_ac.unsqueeze(0))
        else:
            reconstruction_algorithm = OSEM(likelihood)   
            recon_ac_44= reconstruction_algorithm(n_iters=4, n_subsets=4)
            reconstruction_algorithm = OSEM(likelihood)   
            recon_ac_66= reconstruction_algorithm(n_iters=6, n_subsets=6)
            reconstruction_algorithm = OSEM(likelihood)   
            recon_ac_88= reconstruction_algorithm(n_iters=8, n_subsets=8)
            recon_ac_44[:, :, ~mask_atm] = 0
            recon_ac_66[:, :, ~mask_atm] = 0
            recon_ac_44[:, :, ~mask_atm] = 0

            recon_ac_44_batch.append(recon_ac_44.unsqueeze(0))
            recon_ac_66_batch.append(recon_ac_66.unsqueeze(0))
            recon_ac_88_batch.append(recon_ac_88.unsqueeze(0))
    if num_input == 1:
        recon_ac_batch = torch.stack(recon_ac_batch, dim=0).requires_grad_(True)
        return recon_ac_batch
    else:
        recon_ac_44_batch = torch.stack(recon_ac_44_batch, dim=0).requires_grad_(True)
        recon_ac_66_batch = torch.stack(recon_ac_66_batch, dim=0).requires_grad_(True)
        recon_ac_88_batch = torch.stack(recon_ac_88_batch, dim=0).requires_grad_(True)
        return recon_ac_44_batch, recon_ac_66_batch, recon_ac_88_batch
    

def reconstruct_nm_to_nac(raw_file_path, output_dir, colimator='G8-LEHR', energy_keV=140.5):
    """
    Reconstruct NM raw file to NAC images in 3 different OSEM settings
    """
    from pytomography.io.SPECT import dicom
    from pytomography.algorithms import OSEM
    from pytomography.transforms.SPECT import SPECTPSFTransform
    from pytomography.projectors.SPECT import SPECTSystemMatrix
    from pytomography.likelihoods import PoissonLogLikelihood
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "NAC", "OSEM_4I4S"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "NAC", "OSEM_6I6S"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "NAC", "OSEM_8I8S"), exist_ok=True)
    
    # Get metadata and projections
    object_meta, proj_meta = dicom.get_metadata(raw_file_path, index_peak=0)
    photopeak = dicom.get_projections(raw_file_path, index_peak=0)
    psf_meta = dicom.get_psfmeta_from_scanner_params(colimator, energy_keV=energy_keV)
    psf_transform = SPECTPSFTransform(psf_meta)
    # Create system matrix for NAC reconstruction (no attenuation)

    system_matrix_nac = SPECTSystemMatrix(
            obj2obj_transforms=[psf_transform],
            proj2proj_transforms=[],
            object_meta=object_meta,
            proj_meta=proj_meta)

    likelihood = PoissonLogLikelihood(system_matrix_nac, photopeak)
    
    # Reconstruct in 3 different settings
    settings = [
        (4, 4, "OSEM_4I4S"),
        (6, 6, "OSEM_6I6S"), 
        (8, 8, "OSEM_8I8S")
    ]
    
    base_filename = os.path.splitext(os.path.basename(raw_file_path))[0]
    
    for n_iters, n_subsets, setting_name in settings:
        reconstruction_algorithm = OSEM(likelihood)
        recon_nac = reconstruction_algorithm(n_iters=n_iters, n_subsets=n_subsets)

        # Create DICOM output directory for this setting
        dicom_output_dir = os.path.join(output_dir, "NAC", setting_name, "dicom_temp")
        os.makedirs(dicom_output_dir, exist_ok=True)
        
        # Save as DICOM using pytomography
        save_dcm_path = os.path.join(dicom_output_dir, f"{base_filename}")
        dicom.save_dcm(
            save_path=save_dcm_path,
            object=recon_nac,
            file_NM=raw_file_path,
            recon_name='SPECT_NAC',
            single_dicom_file=False
        )
        
        # Convert DICOM to NIfTI
        nifti_output_dir = os.path.join(output_dir, "NAC", setting_name)
        convert_dicom_to_nifti(dicom_output_dir, nifti_output_dir)
        
        # Remove DICOM files
        shutil.rmtree(dicom_output_dir)
        
        print(f"Saved {setting_name}: {nifti_output_dir}")

def get_post_transforms_ac(output_dir, val_transforms_save):
    return Compose([
        Invertd(
            keys="pred",
            transform=val_transforms_save,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        SaveImaged(
            keys="pred",
            meta_keys="pred_meta_dict",
            output_dir=output_dir,
            output_postfix="",
            resample=False,
            separate_folder=False,
        ),
    ])
def apply_post_transforms_ac(batch, output_dir, val_transforms_save):
    post_transforms_ac = get_post_transforms_ac(output_dir, val_transforms_save)
    move_files_up(output_dir, 5)
    return post_transforms_ac([batch])

def move_files_up(parent_dir: str, expected_folders: int) -> None:
    # Get list of subfolders only
    subfolders = [f for f in os.listdir(parent_dir) 
                  if os.path.isdir(os.path.join(parent_dir, f))]
    
    if len(subfolders) != expected_folders:
        print(f"Aborted: Found {len(subfolders)} folders, expected {expected_folders}.")
        return
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(parent_dir, subfolder)
        for file_name in os.listdir(subfolder_path):
            src_path = os.path.join(subfolder_path, file_name)
            dst_path = os.path.join(parent_dir, file_name)
            if os.path.isfile(src_path):
                shutil.move(src_path, dst_path)
        os.rmdir(subfolder_path)
    
    print(f"Moved all files to '{parent_dir}' and removed {expected_folders} folders.")

def get_ac_images_from_models(target_directory, model_files):
    model_image_dict = defaultdict(list)
    for model_name in model_files:
        model_folder = os.path.join(target_directory, model_name)
        if os.path.exists(model_folder):
            image_files = [f for f in os.listdir(model_folder) if f.endswith('.nii.gz')]
            for image in image_files:
                model_image_dict[image].append(os.path.join(model_folder, image))
    return model_image_dict

def ensemble_image_regression(image_paths, target_path):
    import SimpleITK as sitk
    image_ensemble = sitk.ReadImage(image_paths[0])
    for img_path in image_paths[1:]:
        image_ensemble += sitk.ReadImage(img_path)
    image_ensemble = image_ensemble / len(image_paths)
    sitk.WriteImage(image_ensemble, target_path)

def ensemble_across_models(outpath, model_files):
    target_directory = outpath
    os.makedirs(target_directory, exist_ok=True)
    model_image_dict = get_ac_images_from_models(target_directory, model_files)
    for image_name, image_paths in model_image_dict.items():
        if len(image_paths) == len(model_files):
            target_path = os.path.join(target_directory, image_name.replace('_pred', ''))
            ensemble_image_regression(image_paths, target_path)
    # Remove individual model folders after ensemble creation
    print("Removing individual model folders after ensemble creation...")
    for model_name in model_files:
        model_folder = os.path.join(target_directory, model_name)
        if os.path.exists(model_folder):
            import shutil
            shutil.rmtree(model_folder)
            print(f"Removed model folder: {model_folder}")
            
# Function to convert DICOM to NIfTI using SimpleITK
def convert_dicom_to_nifti(input_dir, output_dir):
    import SimpleITK as sitk
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        if files:
            # Assume all files in the current directory are part of one volume
            series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
            if not series_IDs:
                continue

            for series_ID in series_IDs:
                dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, series_ID)
                image = sitk.ReadImage(dicom_names)

                # Create output file path
                folder_name = os.path.basename(root)
                output_file = os.path.join(output_dir, f"{folder_name}.nii.gz")

                # Write NIfTI file
                sitk.WriteImage(image, output_file)
                print(f"Converted {root} to {output_file}")
