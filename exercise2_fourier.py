# imports
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Ανάγνωση εικόνας ως grayscale
image_path = "images-project-1/cornfield.jpg"
image = ski.io.imread(image_path, as_gray=True)

# Α: Υπολογισμός του 2Δ DFT και εμφάνιση φάσματος πλάτους και φάσης
F = fft2(image)
F_shifted = fftshift(F) # μετατόπιση ώστε οι χαμηλές συχνότητες να είναι στο κέντρο

magnitude = np.abs(F_shifted)
phase = np.angle(F_shifted)

# Β: Αντιστροφή του φάσματος φάσης κατά τον κατακόρυφο άξονα
phase_flipped = np.flipud(phase)

# Δημιουργία νέου φάσματος με το αρχικό πλάτος και τη "συμμετρική" φάση
F_modified = magnitude * np.exp(1j * phase_flipped)

# Γ: Υπολογισμός της αντίστροφης DFT και προβολή της εικόνας
F_modified_unshifted = ifftshift(F_modified)
image_modified = ifft2(F_modified_unshifted).real

fig, axs = plt.subplots(2, 3, figsize=(18, 10))

axs[0, 0].imshow(image, cmap="gray")
axs[0, 0].set_title("Αρχική εικόνα")
axs[0, 0].axis("off")

axs[0, 1].imshow(np.log(1 + magnitude), cmap="gray")
axs[0, 1].set_title("Φάσμα Πλάτους (log)")
axs[0, 1].axis("off")

axs[0, 2].imshow(phase, cmap="gray")
axs[0, 2].set_title("Φάσμα Φάσης")
axs[0, 2].axis("off")

axs[1, 0].imshow(np.log(1 + magnitude), cmap="gray")
axs[1, 0].set_title("Πλάτος (με τροποποιημένη φάση)")
axs[1, 0].axis("off")

axs[1, 1].imshow(phase_flipped, cmap="gray")
axs[1, 1].set_title("Τροποποιημένο Φάσμα Φάσης")
axs[1, 1].axis("off")

axs[1, 2].imshow(image_modified, cmap="gray")
axs[1, 2].set_title("Τροποποιημένη Εικόνα")
axs[1, 2].axis("off")

plt.tight_layout()
plt.show()
