import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2, axis=(0, 1, 2))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True, channel_axis=2)

def main():
    # Directory containing the images
    directory = '../../renders'

    global_psnr_orig = 0
    global_ssim_orig = 0

    global_psnr_lod = 0
    global_ssim_lod = 0

    for i in tqdm(range(301)):
        # Format the image index
        img_index = str(i).zfill(5)

        # Load the images
        ref_img_path = os.path.join(directory, f'reference/{img_index}.png')
        orig_img_path = os.path.join(directory, f'orig/{img_index}.png')
        lod_img_path = os.path.join(directory, f'inria/{img_index}.png')

        ref_img = cv2.imread(ref_img_path)
        orig_img = cv2.imread(orig_img_path)
        lod_img = cv2.imread(lod_img_path)

        # Ensure the images are in the same size
        ref_img    = cv2.resize(ref_img,    (980, 545))
        orig_img = cv2.resize(orig_img, (980, 545))
        lod_img = cv2.resize(lod_img, (980, 545))

        psnr = calculate_psnr(ref_img, orig_img)
        ssim = calculate_ssim(ref_img, orig_img)
        global_psnr_orig = global_psnr_orig + psnr
        global_ssim_orig = global_ssim_orig + ssim

        psnr = calculate_psnr(ref_img, lod_img)
        ssim = calculate_ssim(ref_img, lod_img)
        global_psnr_lod = global_psnr_lod + psnr
        global_ssim_lod = global_ssim_lod + ssim

    global_psnr_orig = global_psnr_orig / 301
    global_ssim_orig = global_ssim_orig / 301

    global_psnr_lod = global_psnr_lod / 301
    global_ssim_lod = global_ssim_lod / 301

    print(f'Original renders stats: PSNR: {global_psnr_orig} / SSIM: {global_ssim_orig}')
    print(f'LOD      renders stats: PSNR: {global_psnr_lod} / SSIM: {global_ssim_lod}')

if __name__ == "__main__":
    main()
