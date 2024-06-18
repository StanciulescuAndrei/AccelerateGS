import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm

def calculate_psnr(img1, img2, mask):
    mse = np.sum((img1 * mask - img2 * mask) ** 2, axis=(0, 1, 2))
    mse = mse / np.sum(mask, axis=(0, 1, 2))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True, channel_axis=2)

def main():
    # Directory containing the images
    directory = '../../renders'

    tests = ['octree', 'bvh', 'hybrid_cluster', 'test']
    # tests = ['test']

    for test in tests:

        global_psnr_lod_fg = 0
        global_psnr_lod_bg = 0
        global_ssim_lod = 0

        n_images = 185
        depth_threshold = 68

        for i in tqdm(range(0, n_images, 10)):
            # Format the image index
            img_index = str(i).zfill(5)

            # Load the images
            ref_img_path = os.path.join(directory, f'orig/{img_index}.png')
            lod_img_path = os.path.join(directory, f'{test}/{img_index}.png')
            depth_img_path = os.path.join(directory, f'{test}/d{img_index}.png')

            ref_img = cv2.imread(ref_img_path)
            lod_img = cv2.imread(lod_img_path)
            depth_img = cv2.imread(depth_img_path)

            # Ensure the images are in the same size
            resize_dim = (1600, 1036)
            ref_img  = cv2.resize(ref_img,  resize_dim)
            lod_img  = cv2.resize(lod_img,  resize_dim)
            depth_img  = cv2.resize(depth_img,  resize_dim)

            # SSIM 
            ssim = calculate_ssim(ref_img, lod_img)
            global_ssim_lod = global_ssim_lod + ssim

            # PSNR foreground
            psnr = calculate_psnr(ref_img, lod_img, depth_img < depth_threshold)
            global_psnr_lod_fg = global_psnr_lod_fg + psnr

            # PSNR background
            psnr = calculate_psnr(ref_img, lod_img, depth_img > depth_threshold)
            global_psnr_lod_bg = global_psnr_lod_bg + psnr
            

        global_ssim_lod = global_ssim_lod / 19
        
        global_psnr_lod_bg = global_psnr_lod_bg / 19
        global_psnr_lod_fg = global_psnr_lod_fg / 19
        
        print(f'Renders {test} stats: PSNR FG: {global_psnr_lod_fg} / PSNR BG: {global_psnr_lod_bg} / SSIM: {global_ssim_lod}')

if __name__ == "__main__":
    main()
