# imports
import matplotlib.pyplot as plt
from skimage import io, filters, segmentation, measure, morphology
from skimage.util import img_as_float

# Φόρτωση εικόνας
image = img_as_float(io.imread("images-project-1/parking-lot.jpg"))

# Υπολογισμός gradient (sobel) για εισαγωγή στο watershed
gradient = filters.sobel(image)

# Καθορισμός markers με μορφολογικό άνοιγμα
markers = morphology.label(image < 0.4) # χαμηλά επίπεδα έντασης που υποδηλώνουν πιθανά όρια αντικειμένων

# Εφαρμογή watershed αλγορίθμου
segments = segmentation.watershed(gradient, markers)

# Δημιουργία bounding boxes
regions = measure.regionprops(segments)

# Εμφάνιση αποτελεσμάτων
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image, cmap='gray')

# Σχεδίαση bounding boxes για κάθε ανιχνευμένο αντικείμενο
for region in regions:
    if region.area > 100: # αγνοεί τις πολύ μικρές περιοχές
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             edgecolor='magenta', facecolor='none', linewidth=1.5)
        ax.add_patch(rect)

ax.set_title("Τμηματοποίηση με Watershed")
ax.axis("off")
plt.tight_layout()
plt.show()
