import numpy as np
from monai.networks.nets import SwinUNETR
import os
import torch
from monai.metrics import SSIMMetric, PSNRMetric

from utils import (
    absolute_percent_error_metric,
    nmae,
    nmse,
    spect_ac_recon,
    apply_post_transforms_ac,
)


def calculate_metrics(val_ac, recon_ac_batch, metrics_dict, key_prefix, loss_function, device):
    ssim_metric_ac = SSIMMetric(data_range=val_ac.max(), spatial_dims=3)
    psnr_metric = PSNRMetric(max_val=val_ac.max())
    metrics_dict[f"{key_prefix}_mae"].append(loss_function(recon_ac_batch, val_ac).item())
    metrics_dict[f"{key_prefix}_ssim"].extend(
        ssim_metric_ac(val_ac.to(device), recon_ac_batch).cpu().numpy().flatten()
    )
    mse_ac = torch.nn.functional.mse_loss(recon_ac_batch, val_ac).item()
    metrics_dict[f"{key_prefix}_mse"].append(mse_ac)
    psnr_value = psnr_metric(val_ac, recon_ac_batch).cpu().numpy().flatten()
    metrics_dict[f"{key_prefix}_psnr"].extend(psnr_value)
    metrics_dict[f"{key_prefix}_rae"].extend(
        absolute_percent_error_metric(recon_ac_batch, val_ac).flatten()
    )
    metrics_dict[f"{key_prefix}_nmae"].append(nmae(recon_ac_batch, val_ac))
    metrics_dict[f"{key_prefix}_nmse"].append(nmse(recon_ac_batch, val_ac))


def calculate_metrics_final(model_name, val_ac, recon_ac_batch, metrics_dict, key_prefix, loss_function, device):
    ssim_metric_ac = SSIMMetric(data_range=val_ac.max(), spatial_dims=3)
    psnr_metric = PSNRMetric(max_val=val_ac.max())
    metrics_dict[f"{key_prefix}_mae"][model_name].append(
        loss_function(recon_ac_batch, val_ac).item()
    )
    metrics_dict[f"{key_prefix}_ssim"][model_name].extend(
        ssim_metric_ac(val_ac.to(device), recon_ac_batch).cpu().numpy().flatten()
    )
    mse_ac = torch.nn.functional.mse_loss(recon_ac_batch, val_ac).item()
    metrics_dict[f"{key_prefix}_mse"][model_name].append(mse_ac)
    psnr_value = psnr_metric(val_ac, recon_ac_batch).cpu().numpy().flatten()
    metrics_dict[f"{key_prefix}_psnr"][model_name].extend(psnr_value)
    metrics_dict[f"{key_prefix}_rae"][model_name].extend(
        absolute_percent_error_metric(recon_ac_batch, val_ac).flatten()
    )
    metrics_dict[f"{key_prefix}_nmae"][model_name].append(nmae(recon_ac_batch, val_ac))
    metrics_dict[f"{key_prefix}_nmse"][model_name].append(nmse(recon_ac_batch, val_ac))


def save_best_model(metric_type, metric_value, input_type, best_metrics, best_metric_epochs, epoch, model, outpath, fold):
    metric_value = np.mean(metric_value[f"{input_type}_{metric_type}"])
    if metric_type in ["ssim", "psnr"]:
        if metric_value > best_metrics[input_type][metric_type]:
            best_metrics[input_type][metric_type] = metric_value
            best_metric_epochs[input_type][metric_type] = epoch + 1
            torch.save(
                model.state_dict(),
                os.path.join(outpath, f"SwinUNETR_Fold{fold}_{input_type}_{metric_type}.pt"),
            )
    else:
        if metric_value < best_metrics[input_type][metric_type]:
            best_metrics[input_type][metric_type] = metric_value
            best_metric_epochs[input_type][metric_type] = epoch + 1
            torch.save(
                model.state_dict(),
                os.path.join(outpath, f"SwinUNETR_Fold{fold}_{input_type}_{metric_type}.pt"),
            )


def evaluate_model(
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
                    recon_ac_batch = spect_ac_recon(val_raw, output, 1, num_input, input_type, val_labels, args.colimator)
                    calculate_metrics_final(
                        model_name, val_ac, recon_ac_batch, metrics, input_type, loss_function, device
                    )
                    batch = {"image": val_inputs, "label": val_labels, "pred": recon_ac_batch}
                else:
                    calculate_metrics_final(
                        model_name, val_ac, output, metrics, input_type, loss_function, device
                    )
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
                    ac_44 = val_data["ac44"].to(device)
                    ac_66 = val_data["ac66"].to(device)
                    ac_88 = val_data["ac88"].to(device)
                    raw = val_data["raw"]
                    val_labels = val_data["label"].to(device)
                    output = model(val_inputs)
                    recon_ac_44_batch, recon_ac_66_batch, recon_ac_88_batch = spect_ac_recon(
                        raw, output, 1, num_input, input_type, val_labels, args.colimator
                    )
                    calculate_metrics_final(
                        model_name, ac_44, recon_ac_44_batch, metrics, 'OSEM_4I4S', loss_function, device
                    )
                    calculate_metrics_final(
                        model_name, ac_66, recon_ac_66_batch, metrics, 'OSEM_6I6S', loss_function, device
                    )
                    calculate_metrics_final(
                        model_name, ac_88, recon_ac_88_batch, metrics, 'OSEM_8I8S', loss_function, device
                    )
                    prediction_dir_ac_44 = outpath + "/predictions/OSEM_4I4S"
                    os.makedirs(prediction_dir_ac_44, exist_ok=True)
                    prediction_dir_ac_66 = outpath + "/predictions/OSEM_6I6S"
                    os.makedirs(prediction_dir_ac_66, exist_ok=True)
                    prediction_dir_ac_88 = outpath + "/predictions/OSEM_8I8S"
                    os.makedirs(prediction_dir_ac_88, exist_ok=True)
                    batch = {"image": val_inputs, "label": val_labels, "pred": recon_ac_44_batch}
                    for key in batch:
                        batch[key] = batch[key][0]
                    os.makedirs(os.path.join(prediction_dir_ac_44, model_name), exist_ok=True)
                    apply_post_transforms_ac(batch, os.path.join(prediction_dir_ac_44, model_name), val_transforms_save)
                    batch = {"image": val_inputs, "label": val_labels, "pred": recon_ac_66_batch}
                    for key in batch:
                        batch[key] = batch[key][0]
                    os.makedirs(os.path.join(prediction_dir_ac_66, model_name), exist_ok=True)
                    apply_post_transforms_ac(batch, os.path.join(prediction_dir_ac_66, model_name), val_transforms_save)
                    batch = {"image": val_inputs, "label": val_labels, "pred": recon_ac_88_batch}
                    for key in batch:
                        batch[key] = batch[key][0]
                    os.makedirs(os.path.join(prediction_dir_ac_88, model_name), exist_ok=True)
                    apply_post_transforms_ac(batch, os.path.join(prediction_dir_ac_88, model_name), val_transforms_save)
            if args.indirect:
                output = torch.clamp(output, min=0)
                calculate_metrics_final(model_name, val_labels, output, metrics, 'ATM', loss_function, device)
                batch = {"image": val_inputs, "label": val_labels, "pred": output}
                for key in batch:
                    batch[key] = batch[key][0]
                os.makedirs(os.path.join(prediction_dir_atm, model_name), exist_ok=True)
                apply_post_transforms_ac(batch, os.path.join(prediction_dir_atm, model_name), val_transforms_save)
    return metrics


