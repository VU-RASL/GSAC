import os
import cv2
import numpy as np
import torch
import lpips
from glob import glob
from skimage.metrics import structural_similarity as ssim_metric

###############################################################################
# Helper Functions
###############################################################################

def compute_psnr(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))

def compute_ssim(gt, pred):
    # If the images have shape (H,W,3):
    s = ssim_metric(gt, pred, data_range=1.0, multichannel=True)
    return s

def compute_lpips(gt, pred, loss_fn):
    # shape => (H,W,3) => (3,H,W)
    gt_torch   = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0)
    pred_torch = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0)

    # scale [0..1] -> [-1..1]
    gt_torch   = (gt_torch*2.0 - 1.0).float()
    pred_torch = (pred_torch*2.0 - 1.0).float()

    with torch.no_grad():
        dist = loss_fn(gt_torch, pred_torch)
    return dist.mean().item()

###############################################################################
# Main function
###############################################################################
def main():
    ground_truth_folder = "data/PeopleSnapshot_SMPLX/male-4-casual-test/images/"
    predicted_folder = "logs/gaussians_docker_custom/data-2025_01-11_10-42-test/test/rasterization/"
    # haha's result
    #predicted_folder  = "data/PeopleSnapshot/male-4-casual-black/"
    ground_truth_mask_folder = "data/PeopleSnapshot_SMPLX/male-4-casual-test/masks/"
    output_folder = "output_pred_masks/"

    # Gather all ground-truth images
    gt_files = sorted(glob(os.path.join(ground_truth_folder, "*.png")))
    print(gt_files)
    # Setup LPIPS
    loss_fn = lpips.LPIPS(net='vgg')  # CPU version; use .cuda() if GPU

    psnr_vals = []
    ssim_vals = []
    lpips_vals = []

    for gt_file in gt_files:
        # Example: gt_file = "ground_truth/00000.png"
        filename = os.path.basename(gt_file)   # "00000.png"
        print(filename)
        base_no_ext = os.path.splitext(filename)[0]  # "00000"

        # Convert "00000" -> integer 0
        gt_index = int(base_no_ext)

        # Predict file is "1.png" if gt_index=0
        pred_index = gt_index + 1
        pred_file = os.path.join(predicted_folder, f"{pred_index}.png")
        #pred_file = os.path.join(predicted_folder, f"{pred_index}.png")

        # Mask file is "00000_gray.png"
        mask_file = os.path.join(ground_truth_mask_folder, f"{base_no_ext}_gray.png")

        # Check existence
        if not os.path.exists(pred_file):
            print(f"[Warning] Missing predicted image: {pred_file}, skipping.")
            continue
        if not os.path.exists(mask_file):
            print(f"[Warning] Missing mask file: {mask_file}, skipping.")
            continue

        # Load ground truth image
        gt_img = cv2.imread(gt_file, cv2.IMREAD_COLOR)
        H, W, _ = gt_img.shape
        vertical_offset = (H - W) // 2
        gt_img = gt_img[vertical_offset:vertical_offset + W]
        gt_img = cv2.resize(gt_img, (544, 544))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img = gt_img.astype(np.float32) / 255.0  # => [0..1], shape (H,W,3)

        # Load predicted image
        pred_img = cv2.imread(pred_file, cv2.IMREAD_COLOR)
        if pred_img is None:
            print(f"[Warning] Could not load {pred_file}, skipping.")
            continue
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        pred_img = pred_img.astype(np.float32) / 255.0
   
 
        # Load mask
        mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask_img = mask_img[vertical_offset:vertical_offset + W]
        mask_img = cv2.resize(mask_img, (544, 544))
        mask = (mask_img > 127).astype(np.float32)  # shape (H,W)
        # Expand to 3 channels
        mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Check shape consistency
        if gt_img.shape != pred_img.shape or gt_img.shape != mask_3c.shape:
         
            print(f"[Warning] Shape mismatch for {filename}, skipping.")
            continue

        # Zero out background
        gt_masked   = gt_img   * mask_3c
        pred_masked = pred_img 
        pred_masked_bgr = (pred_masked * 255).astype(np.uint8)  # Convert back to 0-255 range
        pred_masked_bgr = cv2.cvtColor(pred_masked_bgr, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(output_folder, f"{base_no_ext}_pred_mask.png")
        cv2.imwrite(save_path, gt_img)

        # Compute metrics
        cur_psnr  = compute_psnr(gt_masked, pred_masked)
        cur_ssim  = 0
        cur_lpips = compute_lpips(gt_masked, pred_masked, loss_fn)

        psnr_vals.append(cur_psnr)
        ssim_vals.append(cur_ssim)
        lpips_vals.append(cur_lpips)

        print(f"{filename} => Pred {pred_index}.png: "
              f"PSNR={cur_psnr:.3f}, SSIM={cur_ssim:.3f}, LPIPS={cur_lpips:.3f}")

    # Averages
    if psnr_vals:
        mean_psnr  = np.mean(psnr_vals)
        mean_ssim  = np.mean(ssim_vals)
        mean_lpips = np.mean(lpips_vals)
    else:
        mean_psnr  = 0
        mean_ssim  = 0
        mean_lpips = 0

    print("\n========== Final Averages ==========")
    print(f"PSNR:  {mean_psnr:.3f}")
    print(f"SSIM:  {mean_ssim:.3f}")
    print(f"LPIPS: {mean_lpips:.3f}")


if __name__ == "__main__":
    main()
