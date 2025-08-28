# imports
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.feature import peak_local_max
from scipy.ndimage import median_filter, gaussian_filter

# Συναρτήσεις Φιλτραρίσματος ανάλογα με τον θόρυβο

def filter_gaussian_noise(image, sigma=2.95):
    return gaussian_filter(image, sigma=sigma)

def filter_salt_pepper_noise(image, size=3):
    return median_filter(image, size=size)

def filter_periodic_noise(image, notch_radius=12, sigma_blur=1):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log1p(np.abs(fshift))

    # Εντοπισμός περιοχών με περιοδικότητα
    coordinates = peak_local_max(magnitude_spectrum, min_distance=20, threshold_rel=0.5)
    coordinates = [pt for pt in coordinates if np.linalg.norm(np.array(pt) - np.array([crow, ccol])) > 30]

    # Notch mask
    notch_mask = np.ones_like(fshift, dtype=np.float32)
    rr, cc = np.ogrid[:rows, :cols]

    for y, x in coordinates:
        mask = ((rr - y)**2 + (cc - x)**2 <= notch_radius**2)
        notch_mask[mask] = 0
        # συμμετρικό σημείο
        y_sym = crow - (y - crow)
        x_sym = ccol - (x - ccol)
        mask_sym = ((rr - y_sym)**2 + (cc - x_sym)**2 <= notch_radius**2)
        notch_mask[mask_sym] = 0

    # εφαρμόζω notch mask
    f_filtered = fshift * notch_mask
    f_ishift = np.fft.ifftshift(f_filtered)
    img_filtered = np.abs(np.fft.ifft2(f_ishift))
    img_filtered = gaussian_filter(img_filtered, sigma=sigma_blur)

    return img_filtered, magnitude_spectrum, coordinates

# Συνάρτηση SSIM

def compute_ssim(original, denoised):
    return ssim(original, denoised, data_range=1.0)

# Φόρτωση Εικόνων

def load_images():
    original = img_as_float(io.imread("images-project-1/lenna.jpg", as_gray=True))
    n1 = img_as_float(io.imread("images-project-1/lenna-n1.jpg", as_gray=True))
    n2 = img_as_float(io.imread("images-project-1/lenna-n2.jpg", as_gray=True))
    n3 = img_as_float(io.imread("images-project-1/lenna-n3.jpg", as_gray=True))
    return original, n1, n2, n3

# Main συνάρτηση

def main():
    original, n1, n2, n3 = load_images()

    # φιλτράρισμα
    n1_filtered = filter_gaussian_noise(n1)
    n2_filtered = filter_salt_pepper_noise(n2)
    n3_filtered, spectrum, peaks = filter_periodic_noise(n3, notch_radius=12)

    # SSIM
    ssim_n1 = compute_ssim(original, n1_filtered)
    ssim_n2 = compute_ssim(original, n2_filtered)
    ssim_n3 = compute_ssim(original, n3_filtered)

    # διαγράμματα
    fig, axs = plt.subplots(4, 2, figsize=(12, 14))

    axs[0, 0].imshow(n1, cmap='gray')
    axs[0, 0].set_title("lenna-n1 (Gaussian noise)")
    axs[0, 1].imshow(n1_filtered, cmap='gray')
    axs[0, 1].set_title(f"Filtered (SSIM: {ssim_n1:.4f})")

    axs[1, 0].imshow(n2, cmap='gray')
    axs[1, 0].set_title("lenna-n2 (Salt & Pepper noise)")
    axs[1, 1].imshow(n2_filtered, cmap='gray')
    axs[1, 1].set_title(f"Filtered (SSIM: {ssim_n2:.4f})")

    axs[2, 0].imshow(n3, cmap='gray')
    axs[2, 0].set_title("lenna-n3 (Periodic noise)")
    axs[2, 1].imshow(n3_filtered, cmap='gray')
    axs[2, 1].set_title(f"Filtered (SSIM: {ssim_n3:.4f})")

    axs[3, 0].imshow(spectrum, cmap='gray')
    axs[3, 0].set_title("Magnitude Spectrum")
    axs[3, 0].scatter([x[1] for x in peaks], [x[0] for x in peaks], color='yellow', s=10)
    axs[3, 1].axis("off")

    for ax in axs.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# **** main ****
if __name__ == "__main__":
    main()
