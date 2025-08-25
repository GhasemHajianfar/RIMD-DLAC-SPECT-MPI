import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk

from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from termcolor import cprint


def path_join(url, start_index, stop_index):
    if sys.platform == "win32":
        if start_index == -1:
            path_updated = os.path.basename(url)
        else:
            path_updated = "\\".join(url.split("\\")[start_index:stop_index])
    elif sys.platform.lower() == "linux":
        if start_index == -1:
            path_updated = os.path.basename(url)
        else:
            path_updated = "/".join(url.split("/")[start_index:stop_index])
    return path_updated   

def ssim(predicted, reference, minimum = -1000, maximum = 2000, data_range = 1):    
    if isinstance(predicted, str):
        predicted_array = sitk.GetArrayFromImage(sitk.ReadImage(predicted))
    elif isinstance(predicted, sitk.Image):
        predicted_array = sitk.GetArrayFromImage(predicted)
    elif isinstance(predicted, np.ndarray):
        predicted_array = predicted
        
    if isinstance(reference, str):
        reference_array = sitk.GetArrayFromImage(sitk.ReadImage(reference))
    elif isinstance(reference, sitk.Image):
        reference_array = sitk.GetArrayFromImage(reference)
    elif isinstance(reference, np.ndarray):
        reference_array = reference
    #clipping    
    predicted_array[predicted_array < minimum] = minimum
    predicted_array[predicted_array > maximum] = maximum
    reference_array[reference_array < minimum] = minimum
    reference_array[reference_array > maximum] = maximum
    # normalizing
    predicted_array = (predicted_array - np.min(predicted_array)) / (np.max(predicted_array) - np.min(predicted_array))
    reference_array = (reference_array - np.min(reference_array)) / (np.max(reference_array) - np.min(reference_array))
    
    
    ssim_value = sk_ssim(predicted_array, reference_array, data_range = data_range)
    return ssim_value, ssim_value * 100

def psnr(predicted, reference, minimum = -1000, maximum = 2000, data_range = 1):
    if isinstance(predicted, str):
        predicted_array = sitk.GetArrayFromImage(sitk.ReadImage(predicted))
    elif isinstance(predicted, sitk.Image):
        predicted_array = sitk.GetArrayFromImage(predicted)
    elif isinstance(predicted, np.ndarray):
        predicted_array = predicted
        
    if isinstance(reference, str):
        reference_array = sitk.GetArrayFromImage(sitk.ReadImage(reference))
    elif isinstance(reference, sitk.Image):
        reference_array = sitk.GetArrayFromImage(reference)
    elif isinstance(reference, np.ndarray):
        reference_array = reference
    #clipping    
    predicted_array[predicted_array < minimum] = minimum
    predicted_array[predicted_array > maximum] = maximum
    reference_array[reference_array < minimum] = minimum
    reference_array[reference_array > maximum] = maximum
    # normalizing
    predicted_array = (predicted_array - np.min(predicted_array)) / (np.max(predicted_array) - np.min(predicted_array))
    reference_array = (reference_array - np.min(reference_array)) / (np.max(reference_array) - np.min(reference_array))
    
    
    psnr_value = sk_psnr(predicted_array, reference_array, data_range = data_range)
    return psnr_value 
        
def percent_error(reference_image, predicted_image, lower_treshold = "none", upper_treshold = "none"):
    if torch.is_tensor(reference_image):
        reference_image = reference_image.cpu().detach().numpy()
    if torch.is_tensor(predicted_image):
        predicted_image = predicted_image.cpu().detach().numpy()
    if lower_treshold != "none":
        reference_image[reference_image<lower_treshold] = np.min(reference_image)
        predicted_image[predicted_image<lower_treshold] = np.min(predicted_image)
    if upper_treshold != "none":
        reference_image[reference_image > upper_treshold] = np.max(reference_image)
        predicted_image[predicted_image > upper_treshold] = np.max(predicted_image)
            
    bias_map = predicted_image - reference_image
    with np.errstate(divide='ignore', invalid='ignore'):
        re_percent_map = (bias_map / reference_image) * 100
        rae_percent_map = abs(bias_map / reference_image) * 100
    
    re_percent = np.mean(np.ma.masked_invalid(re_percent_map[re_percent_map!=0]))
    rae_percent = np.mean(np.ma.masked_invalid(rae_percent_map[re_percent_map!=0]))
    return re_percent, rae_percent, bias_map, re_percent_map, rae_percent_map


def CopyInfo(ReferenceImage, UpdatingImage, origin = True, spacing = True, direction = True):
    if isinstance(ReferenceImage, str):
        ReferenceImage = sitk.ReadImage(ReferenceImage)
    if isinstance(UpdatingImage, str):
        UpdatingImage = sitk.ReadImage(UpdatingImage)
    UpdatedImage = UpdatingImage 
    if origin:
        UpdatedImage.SetOrigin(ReferenceImage.GetOrigin())
    if spacing:
        UpdatedImage.SetSpacing(ReferenceImage.GetSpacing())
    if direction:
        UpdatedImage.SetDirection(ReferenceImage.GetDirection())
    return UpdatedImage

def match_space(input_image, reference_image, interpolate = "linear", DefaultPixelValue = 0, copy_info_prior_to_macth = False):
    if isinstance(input_image, str):
        input_image = sitk.ReadImage(input_image)
        if copy_info_prior_to_macth:
            input_image = CopyInfo(ReferenceImage = reference_image, UpdatingImage = input_image)
    if isinstance(reference_image, str):
        reference_image = sitk.ReadImage(reference_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetOutputDirection(reference_image.GetDirection())
    # Set the default pixel value to -1000
    resampler.SetDefaultPixelValue(DefaultPixelValue)
    if interpolate == "linear":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolate == "nearest":
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolate.lower() == "bspline":
        resampler.SetInterpolator(sitk.sitkBSpline)
    resampled_image = resampler.Execute(input_image)
    return resampled_image


def image_match_eval(predicted,
                        reference,
                        list_of_segments  = "none", 
                        minimum = 0, maximum = 1000000000, 
                        ssim_data_range = 1, 
                        comment = "Calculations", 
                        folds_number = "NotInformed", 
                        body_contour = "none", force_match = False,
                        force_macth_to = "reference", # can be "reference", or "predicted"
                        segmetn_name_path_level = -1,
                        verbose = False,
                        ):
   
    warnings.filterwarnings("ignore")
    tensor_calculation = False
    
    whole_image_df = pd.DataFrame()
    if isinstance(reference, torch.Tensor) or isinstance(predicted, torch.Tensor):
        tensor_calculation = True
    if isinstance(reference, str):
        reference_image = sitk.ReadImage(reference)
        reference_array = sitk.GetArrayFromImage(reference_image)
    elif isinstance(reference, sitk.Image):
        reference_image = reference
        reference_array = sitk.GetArrayFromImage(reference_image)
    elif isinstance(reference, torch.Tensor):
        reference_array = reference.detach().cpu().numpy()


    if isinstance(predicted, str):
        predicted_image = sitk.ReadImage(predicted)
        predicted_array = sitk.GetArrayFromImage(predicted_image)
    elif isinstance(predicted, sitk.Image):
        predicted_image = predicted
        predicted_array = sitk.GetArrayFromImage(predicted_image)
    elif isinstance(predicted, torch.Tensor):
        predicted_array = predicted.detach().cpu().numpy()
    
    if force_match:
        if force_macth_to == "reference":
            predicted_image = match_space(input_image = predicted_image, reference_image = reference)
            predicted_array = sitk.GetArrayFromImage(predicted_image)
            if verbose:
                cprint(f"matched space by force: {predicted} to REFERENCE image", "yellow", "on_cyan")
        elif force_macth_to == "predicted":
            reference_image = match_space(input_image = reference_image, reference_image = predicted_image)
            reference_array = sitk.GetArrayFromImage(reference_image)

            # cprint(f"matched space by force: {predicted} to PREDICTED image", "yellow", "on_cyan")


    
    
    # croppping and masking by bodycontours
    if body_contour != "none":
        if isinstance(body_contour, str):
            body_contour_array = sitk.GetArrayFromImage(match_space(body_contour, reference_image))
        elif isinstance(body_contour, sitk.Image):
            body_contour_array = sitk.GetArrayFromImage(body_contour)
        
        reference_array[body_contour_array < 1] = minimum
        predicted_array[body_contour_array < 1] = minimum

    #clipping    
    predicted_array[predicted_array < minimum] = minimum
    predicted_array[predicted_array > maximum] = maximum
    reference_array[reference_array < minimum] = minimum
    reference_array[reference_array > maximum] = maximum
    # BiasMaps
    bias_map = predicted_array - reference_array
    with np.errstate(divide='ignore', invalid='ignore'):
        percent_bias_map = (bias_map / reference_array) * 100
    bias_map_to_save = bias_map
    bias_map_to_save[np.isnan(bias_map_to_save)] = 0
    bias_map_to_save[np.isinf(bias_map_to_save)] = 0
    if not tensor_calculation:
        bias_map_to_save_image = CopyInfo(reference_image, sitk.GetImageFromArray(bias_map_to_save))
    else:
        bias_map_to_save_image = "Not For Tensor and arrays"
    
    percent_bias_map_to_save = percent_bias_map
    percent_bias_map_to_save[np.isnan(percent_bias_map_to_save)] = 0
    percent_bias_map_to_save[np.isinf(percent_bias_map_to_save)] = 0
    if not tensor_calculation:
        percent_bias_map_to_save_image = CopyInfo(reference_image, sitk.GetImageFromArray(percent_bias_map_to_save))
    else:
        percent_bias_map_to_save_image = "Not For Tensor and arrays"

            

                            
    mean_error = np.mean(np.ma.masked_invalid(bias_map))
    absolute_mean_error = np.mean(np.abs(np.ma.masked_invalid(bias_map)))
    RMSE = np.sqrt(np.mean(mean_error**2))
    MSE = np.mean(mean_error**2)
    re_percent, rae_percent, bias_map, re_percent_map, rae_percent_map = percent_error(reference_array, predicted_array, lower_treshold = "none", upper_treshold = "none")
    if not tensor_calculation:
        SSIM = ssim(predicted_array, reference_array, minimum = minimum, maximum = maximum, data_range = ssim_data_range)[1]
        PSNR = psnr(predicted_array, reference_array, minimum = minimum, maximum = maximum, data_range = ssim_data_range)
    else:
        SSIM = 0
        PSNR = 0

    minimum_value = image_array.min()
    maximum_value = image_array.max()
    range_value = maximum_value - minimum_value
    normalized_mean_abs_error = mean_absolute_error/range_value * 100      
                            
    whole_image_df.at[0, "predicted_url"] = predicted if isinstance(predicted, str) else "Loaded from image"
    whole_image_df.at[0, "reference_url"] = reference if isinstance(reference, str) else "Loaded from image"
    whole_image_df.at[0, "predicted_name"] =  os.path.basename(predicted) if isinstance(predicted, str) else "Loaded from image"
    whole_image_df.at[0, "reference_name"] =  os.path.basename(reference) if isinstance(reference, str) else "Loaded from image"
    whole_image_df.at[0, "model_directory"] = os.path.basename(os.path.dirname(predicted)) if isinstance(predicted, str) else "Loaded from image"
    whole_image_df.at[0, "comment"] = comment
    whole_image_df.at[0, "folds_number"] = folds_number
    whole_image_df.at[0, "minimum-to-calc"] = minimum
    whole_image_df.at[0, "maximum-to-calc"] = maximum

    whole_image_df.at[0, "mean_error"] = mean_error
    whole_image_df.at[0, "absolute_mean_error"] = absolute_mean_error
    whole_image_df.at[0, "normalized_mean_abs_error"] = normalized_mean_abs_error

    whole_image_df.at[0, "RMSE"] = RMSE
    whole_image_df.at[0, "MSE"] = MSE
    whole_image_df.at[0, "re_percent"] = re_percent
    whole_image_df.at[0, "rae_percent"] = rae_percent
    whole_image_df.at[0, "SSIM"] = SSIM
    whole_image_df.at[0, "PSNR"] = PSNR
    if list_of_segments == "none":
        segment_errors_df = "NotSegmentIncluded"
    else:
        segment_errors_df = pd.DataFrame()
        conter_segment_label = 0
        for index, segment_url in enumerate(list_of_segments):
            if isinstance(segment_url, str):
                segment_array = sitk.GetArrayFromImage(match_space(segment_url, reference_image))
                segment_name = path_join(segment_url, segmetn_name_path_level, segmetn_name_path_level+1).replace(".nii.gz", "")
            elif isinstance(segment_url, sitk.Image):
                segment_array = sitk.GetArrayFromImage(segment_url)
                segment_name = "From Image"
            elif isinstance(segment_url, np.ndarray):
                segment_array = segment_url
                segment_name = "From Array"
            elif isinstance(segment_url, torch.Tensor):
                segment_array = segment_url.detach().cpu().numpy()
                segment_name = "From Tensor"
                
            num_unique_labels = list(np.unique(segment_array))
            if 0 in num_unique_labels:
                num_unique_labels.remove(0)
            for label in num_unique_labels:
                predicted_segmented_array = predicted_array[segment_array == label]
                reference_segmented_array = reference_array[segment_array == label]
                biasmap_segmented_array = bias_map[segment_array == label]
                percent_biasmap_segmented_array = re_percent_map[segment_array == label]
                
                segment_errors_df.at[conter_segment_label, "predicted_url"] = predicted if isinstance(predicted, str) else "Loaded from image"
                segment_errors_df.at[conter_segment_label, "reference_url"] = reference if isinstance(reference, str) else "Loaded from image"
                segment_errors_df.at[conter_segment_label, "segment_name_complete"] = segment_name
                
                segment_errors_df.at[conter_segment_label, "predicted_name"] =  os.path.basename(predicted) if isinstance(predicted, str) else "Loaded from image"
                segment_errors_df.at[conter_segment_label, "reference_name"] =  os.path.basename(reference) if isinstance(reference, str) else "Loaded from image"
                segment_errors_df.at[conter_segment_label, "model_directory"] = os.path.basename(os.path.dirname(predicted)) if isinstance(predicted, str) else "Loaded from image"

                segment_errors_df.at[conter_segment_label, "segment_name"] = segment_name.replace((os.path.basename(predicted) if isinstance(predicted, str) else "Loaded from image").replace(".nii.gz", ""), "")
                segment_errors_df.at[conter_segment_label, "Label"] = label
                segment_errors_df.at[conter_segment_label, "comment"] = comment
                segment_errors_df.at[conter_segment_label, "folds_number"] = folds_number
                # refernece
                segment_errors_df.at[conter_segment_label, "Refernce_Mean"] = np.mean(np.ma.masked_invalid(reference_segmented_array))
                segment_errors_df.at[conter_segment_label, "Refernce_Median"] = np.ma.median(np.ma.masked_invalid(reference_segmented_array), axis=0)
                segment_errors_df.at[conter_segment_label, "Refernce_STD"] = np.std(np.ma.masked_invalid(reference_segmented_array))
                # predicted
                segment_errors_df.at[conter_segment_label, "Prediceted_Mean"] = np.mean(np.ma.masked_invalid(predicted_segmented_array))
                segment_errors_df.at[conter_segment_label, "Prediceted_Median"] = np.ma.median(np.ma.masked_invalid(predicted_segmented_array), axis=0)
                segment_errors_df.at[conter_segment_label, "Prediceted_STD"] = np.std(np.ma.masked_invalid(predicted_segmented_array))
                # errors
                # segment_errors_df.at[index, "ME-Voxel"] = np.mean(np.ma.masked_invalid(biasmap_segmented_array))
                # segment_errors_df.at[index, "MAE-Voxel"] = np.mean(np.abs(np.ma.masked_invalid(biasmap_segmented_array)))
                # segment_errors_df.at[index, "RE%_Voxel"] = np.mean(np.ma.masked_invalid(percent_biasmap_segmented_array))
                # segment_errors_df.at[index, "RAE%-Voxel"] = np.mean(np.abs(np.ma.masked_invalid(percent_biasmap_segmented_array)))
                
                segment_errors_df.at[conter_segment_label, "ME-voxel"] = np.mean(np.ma.masked_invalid(predicted_segmented_array)) - np.mean(np.ma.masked_invalid(reference_segmented_array))
                segment_errors_df.at[conter_segment_label, "MAE-voxel"] = np.abs(np.mean(np.ma.masked_invalid(predicted_segmented_array)) - np.mean(np.ma.masked_invalid(reference_segmented_array)))
                segment_errors_df.at[conter_segment_label, "Median-Shift-voxel"] = np.ma.median(np.ma.masked_invalid(predicted_segmented_array), axis=0) - np.ma.median(np.ma.masked_invalid(reference_segmented_array), axis=0)
                segment_errors_df.at[conter_segment_label, "RE%-voxel"] = np.mean(np.ma.masked_invalid(percent_biasmap_segmented_array))
                segment_errors_df.at[conter_segment_label, "RAE%-voxel"] = np.mean(np.abs(np.ma.masked_invalid(percent_biasmap_segmented_array)))
                
                segment_errors_df.at[conter_segment_label, "ME-region"] = (segment_errors_df.at[conter_segment_label, "Prediceted_Mean"] - segment_errors_df.at[conter_segment_label, "Refernce_Mean"])
                segment_errors_df.at[conter_segment_label, "MAE-region"] = abs(segment_errors_df.at[conter_segment_label, "Prediceted_Mean"] - segment_errors_df.at[conter_segment_label, "Refernce_Mean"])
                segment_errors_df.at[int(conter_segment_label), "Median-Shift-region"] = (segment_errors_df.at[int(conter_segment_label), "Prediceted_Median"] - segment_errors_df.at[int(conter_segment_label), "Refernce_Median"])
                segment_errors_df.at[conter_segment_label, "RE%-region"] = 100*(segment_errors_df.at[conter_segment_label, "Prediceted_Mean"] - segment_errors_df.at[conter_segment_label, "Refernce_Mean"])/segment_errors_df.at[conter_segment_label, "Refernce_Mean"]
                segment_errors_df.at[conter_segment_label, "RAE%-region"] = 100*abs(segment_errors_df.at[conter_segment_label, "Prediceted_Mean"] - segment_errors_df.at[conter_segment_label, "Refernce_Mean"])/segment_errors_df.at[conter_segment_label, "Refernce_Mean"]
                
                conter_segment_label += 1
            
            
    return whole_image_df, segment_errors_df, bias_map_to_save_image, percent_bias_map_to_save_image
        
