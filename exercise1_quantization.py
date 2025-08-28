# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.metrics import mean_squared_error


def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB') # χώρος RGB
    return np.array(image)


def quantize_image(img_array, n_colours):
    h, w, c = img_array.shape
    flat_img = img_array.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colours, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(flat_img)
    centroids = kmeans.cluster_centers_.astype('uint8')

    quantized_flat = centroids[labels]
    quantized_img = quantized_flat.reshape(h, w, 3)

    mse = mean_squared_error(flat_img, quantized_flat)
    return quantized_img, mse


def plot_results(original, results, errors, colour_levels):
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(18, 6))

    axes[0].imshow(original)
    axes[0].set_title("Αρχική Εικόνα")
    axes[0].axis("off")

    for i, (quant_img, mse, k) in enumerate(zip(results, errors, colour_levels)):
        axes[i + 1].imshow(quant_img)
        axes[i + 1].set_title(f"K={k}\nMSE={mse:.2f}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


# **** main ****

image_path = "images-project-1/flowers.jpg" # path εικόνας
original_img = load_image(image_path)

# Επίπεδα κβάντισης
colour_levels = [5, 20, 200, 1000]
results = []
errors = []

for k in colour_levels:
    quantized_img, mse = quantize_image(original_img, k)
    results.append(quantized_img)
    errors.append(mse)

# Εμφάνιση αποτελεσμάτων
plot_results(original_img, results, errors, colour_levels)
