# Digital Image Processing and Applications Project_1

## Overview
Developed as a project for the **Digital Image Processing and Applications** course at [Harokopio University of Athens – Dept. of Informatics and Telematics](https://www.dit.hua.gr), this project implements a series of exercises related to image processing, compression, filtering, segmentation, and optimization.
The exercises explore key concepts in computer vision and image analysis, including quantization, Fourier transform, filtering, denoising, and object detection.

All implementations are done in **Python** using libraries such as NumPy, OpenCV, scikit-image, scikit-learn, and Matplotlib.

---

## Exercises

### Exercise 1: Quantization
- **Task A:** Implement a color quantizer in the RGB space using the **K-means clustering algorithm**.
- **Task B:** Generate quantized versions of the image `flowers.jpg` with **5, 20, 200, and 1000 color levels**.
- **Task C:** Compute the **Mean Squared Error (MSE)** for each quantization level and discuss the results.

---

### Exercise 2: Fourier Transform
- **Task A:** Compute the **2D Discrete Fourier Transform (DFT)** of the grayscale image `cornfield.jpg` and display the **magnitude** and **phase spectra**.
- **Task B:** Replace the phase spectrum with its **vertically symmetric version** and recompute the Fourier spectra.
- **Task C:** Compute the **Inverse Fourier Transform** of the modified spectrum and display the resulting image. Provide commentary on the results.

---

### Exercise 3: Filtering
- **Task A:** For the image `girlface.jpg`, design and apply a filter for **object/feature detection**.
- **Task B:** For the image `fruits.jpg`, design and apply a filter for **edge detection** of objects.
- **Task C:** For the image `leaf.jpg`, design and apply a filter to enhance the **fine details** of the objects.

For each case:
- Display the **filtered image**.
- Provide a detailed explanation of the filter design process (e.g., filter type, parameters).

---

### Exercise 4: Optimization
- **Images Used:** `lenna.jpg` (reference image) and noisy versions: `lenna-n1.jpg`, `lenna-n2.jpg`, `lenna-n3.jpg`.
- **Task A:** Design and apply suitable filters for **denoising** each noisy version of the image. Display the best result for each case and explain the filter design choices.
- **Task B:** Compute the **SSIM (Structural Similarity Index Measure)** between the denoised images and the reference image (`lenna.jpg`) and discuss the results.

---

### Exercise 5: Segmentation (Object Detection)
- **Task A:** Design, describe, and implement a segmentation algorithm for detecting objects in the grayscale image `parking-lot.jpg`. For each detected object, compute a **bounding box**.
- **Task B:** Display the original image with all **bounding boxes drawn in a distinct color**. Provide analysis of the results.

---

### Exercise 6: Segmentation (Character Detection)
- **Task A:** Convert the color image `book-cover.jpeg` to grayscale and implement a segmentation algorithm for detecting **characters/letters** in the image.
- **Task B:** Display a **binary segmentation mask** where the foreground corresponds to detected characters and the background corresponds to all other pixels. Discuss the effectiveness of the segmentation.

---

## Technologies Used
- **Python 3**
- **NumPy** – Matrix and numerical operations
- **OpenCV** – Image manipulation and filtering
- **scikit-image** – Image processing utilities
- **scikit-learn** – K-means clustering for quantization
- **Matplotlib** – Visualization
- **tqdm** – Progress bars during dataset/image processing

---

## Results and Discussion
- Quantization shows that increasing the number of colors reduces error but at the cost of storage size.
- Fourier transform manipulation demonstrates the importance of the **phase spectrum** in image reconstruction.
- Filtering enables targeted enhancement such as edge detection, feature extraction, and detail sharpening.
- Optimization through denoising highlights trade-offs between removing noise and preserving fine image details, validated using **SSIM**.
- Segmentation tasks demonstrate object and character detection using thresholding, morphological operations, and bounding box extraction.

---

## How to Run
1. Clone this repository:
   ```bash
   git clone <https://github.com/AthosExarchou/Digital-Image-Processing-and-Applications-Project_1.git>
   cd <Digital-Image-Processing-and-Applications-Project_1>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run each exercise script individually (examples):
   ```bash
   python exercise1_quantization.py
   python exercise2_fourier.py
   python exercise3_filtering.py
   python exercise4_optimization.py
   python exercise5_segmentation.py
   python exercise6_character_segmentation.py
   ```

---

## Author

- **Name**: Exarchou Athos
- **Student ID**: it2022134
- **Email**: it2022134@hua.gr, athosexarhou@gmail.com

## License
This project is licensed under the MIT License.
