# imports
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import sobel_h, sobel_v
from skimage.feature import canny
from scipy.ndimage import gaussian_laplace

fig, axs = plt.subplots(1, 3, figsize=(18, 7))

# Α. girlface.jpg

# Φίλτρο Sobel: ανίχνευση κύριων χαρακτηριστικών προσώπου
image_girl = io.imread("images-project-1/girlface.jpg")

# Υπολογισμός οριζόντιας και κάθετης κλίσης
grad_h = sobel_h(image_girl)
grad_v = sobel_v(image_girl)

# Υπολογισμός μέτρου κλίσης (|G(n,m)|)
gradient_magnitude = np.hypot(grad_h, grad_v)

# Κανονικοποίηση στο [0, 255]
gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255

# Κατώφλι T = 65
T = 65
sobel_thresh = gradient_magnitude > T

axs[0].imshow(sobel_thresh, cmap="gray")
axs[0].set_title("Α. Κύρια στοιχεία (Sobel)")
axs[0].axis("off")

# Β. fruits.jpg

# Φίλτρο Canny: περίγραμμα αντικειμένων
image_fruits = io.imread("images-project-1/fruits.jpg")

# Εφαρμογή φίλτρου Canny με παραμέτρους sigma, T1, T2
sigma = 2.0  # Τυπική απόκλιση για θόλωση
T1 = 5  # Κατώφλι κάτω
T2 = 10  # Κατώφλι πάνω

# Εφαρμογή φίλτρου Canny με κατώφλια και σ
canny_edges = canny(image_fruits, low_threshold=T1, high_threshold=T2, sigma=sigma)

axs[1].imshow(canny_edges, cmap="gray")
axs[1].set_title("Β. Περίγραμμα αντικειμένων (Canny)")
axs[1].axis("off")

# Γ. leaf.jpg

image_leaf = io.imread("images-project-1/leaf.jpg")

# Εφαρμογή LoG με σ=1.5
log_edges = gaussian_laplace(image_leaf, sigma=1.5)

# Υπολογισμός του απόλυτου της εικόνας LoG
log_edges_abs = np.abs(log_edges)

# Κανονικοποίηση της εικόνας ώστε να έχει τιμές μεταξύ 0 και 1
log_edges_normalized = log_edges_abs / np.max(log_edges_abs)

# Εφαρμογή κατωφλίου T=0.7 για να απορριφθούν μικρές τιμές που θεωρούνται θόρυβος
T = 0.7
log_edges_normalized[log_edges_normalized < T] = 0

axs[2].imshow(log_edges_normalized, cmap="gray")
axs[2].set_title("Γ. Λεπτομέρειες αντικειμένων (LoG)")
axs[2].axis("off")

plt.tight_layout()
plt.show()
