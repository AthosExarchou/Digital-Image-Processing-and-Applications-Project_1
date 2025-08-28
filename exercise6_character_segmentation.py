# imports
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, exposure
from skimage.filters import threshold_local

# Φόρτωση και μετατροπή σε grayscale
image = io.imread("images-project-1/book-cover.jpeg")
gray = color.rgb2gray(image)

# Εξισορρόπηση ιστογράμματος για καλύτερη αντίθεση
gray_eq = exposure.equalize_adapthist(gray, clip_limit=0.01)

# Τοπικό κατώφλι για καλύτερο διαχωρισμό χαρακτήρων-αντικειμένων
block_size = 15
local_thresh = threshold_local(gray_eq, block_size=block_size, offset=0.01)
binary = gray_eq < local_thresh

# Καθαρισμός πολύ μικρών αντικειμένων
binary = morphology.remove_small_objects(binary, min_size=10)

# Μορφολογικό κλείσιμο για σύνδεση χαρακτήρων
binary = morphology.binary_closing(binary, morphology.disk(1))

# Εμφάνιση εικόνων
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(gray_eq, cmap='gray')
axes[0].set_title("Γκρι εικόνα")
axes[0].axis('off')

axes[1].imshow(binary, cmap='gray')
axes[1].set_title("Δυαδική μάσκα χαρακτήρων")
axes[1].axis('off')

plt.tight_layout()
plt.show()
