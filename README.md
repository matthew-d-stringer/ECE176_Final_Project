# Image Inpainting for Photo Restoration


## TODO:
1. invert mask colours for 
## Overview
This project aims to develop an image inpainting tool that can restore damaged or missing parts of old photographs using deep learning techniques. By leveraging the method of partial convolutions introduced by Liu, Reda, Shih, and others from Nvidia [1], our model will be capable of producing high-quality restorations with minimal distortions typically found in traditional inpainting methods.

## Motivation
Inspired by personal experiences with old family photographs, this project seeks to create a more accessible and effective solution for restoring sentimental images. Unlike generic off-the-shelf inpainting models, our implementation focuses on:
- Handling various severities of image damage, from minor creases to large missing sections.
- Providing multiple restoration outcomes for users to choose from.
- Enhancing rather than generating new image content, preserving the authenticity of original photos.

## Methodology
### Chosen Approach
We utilize an inpainting method based on Partial Convolutions, where:
- Convolutions are masked and renormalized to improve the accuracy of hole predictions.
- The method ensures hole-filling is independent of initial hole values, minimizing artifacts like blurriness and unnatural edges.
- This approach reduces the need for post-processing, yielding cleaner and more realistic restorations.

### Dataset
We plan to train and test our model using the following datasets:
1. **CelebA** - A dataset of celebrity faces, aiding in reconstructing human portraits.
2. **Places2** - A dataset containing various background scenes, useful for restoring non-portrait images.
3. Additional datasets may be included as needed to improve the model’s adaptability.

The datasets will be used in their standard format, and further research will be conducted on how training on both face and background datasets influences performance in mixed images.

## Experiments
We will conduct experiments to:
- Assess the model’s ability to restore images with different levels of damage.
- Compare results against traditional inpainting techniques.
- Evaluate the impact of training on mixed datasets (faces + backgrounds).
- Generate multiple possible restorations for a given image and analyze user preference.

## References
[1] Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, and Bryan Catanzaro. *Image inpainting for irregular holes using partial convolutions*, 2018.

---
This README provides a high-level overview of our project, including motivation, methodology, datasets, and planned experiments. Further updates will include implementation details and results.
