from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
from skimage.restoration import richardson_lucy
from scipy.signal import convolve2d
import os

from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
from skimage.restoration import richardson_lucy
import os

# app = Flask(__name__)
# UPLOAD_FOLDER = 'static'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # PSF kernel
# def psf_kernel(size=5, sigma=2.0):
#     ax = np.linspace(-(size // 2), size // 2, size)
#     xx, yy = np.meshgrid(ax, ax)
#     kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
#     return kernel / np.sum(kernel)

# # Manual 2D convolution
# def manual_convolve2d(image, kernel):
#     image_h, image_w = image.shape
#     kernel_h, kernel_w = kernel.shape
#     pad_h = kernel_h // 2
#     pad_w = kernel_w // 2

#     padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')
#     output = np.zeros_like(image, dtype=np.float32)

#     for i in range(image_h):
#         for j in range(image_w):
#             region = padded_image[i:i+kernel_h, j:j+kernel_w]
#             output[i, j] = np.sum(region * kernel)

#     return output

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     before_path = None
#     after_path = None

#     if request.method == 'POST':
#         f = request.files['image']
#         before_path = os.path.join(app.config['UPLOAD_FOLDER'], 'before.png')
#         after_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hasil.png')

#         image = Image.open(f).convert('L')
#         image.save(before_path)

#         img = np.array(image).astype(np.float32) / 255.0

#         # Proses gambar
#         psf = psf_kernel(5, 2.0)
#         lucy_restored = richardson_lucy(img, psf, num_iter=10)
#         lucy_scaled = np.clip(lucy_restored * 255, 0, 255).astype(np.uint8)

#         high_pass_kernel = np.array([
#             [-1, -1, -1],
#             [-1,  9, -1],
#             [-1, -1, -1]
#         ])
#         sharpened = manual_convolve2d(lucy_scaled, high_pass_kernel)
#         sharpened_clipped = np.clip(sharpened, 0, 255).astype(np.uint8)

#         gaussian_kernel = np.array([
#             [1, 2, 1],
#             [2, 4, 2],
#             [1, 2, 1]
#         ]) / 16.0
#         blurred_final = manual_convolve2d(sharpened_clipped, gaussian_kernel)
#         blurred_final_clipped = np.clip(blurred_final, 0, 255).astype(np.uint8)

#         G = 1.3
#         P = np.mean(blurred_final_clipped)
#         contrast_enhanced = G * (blurred_final_clipped.astype(np.float32) - P) + P
#         contrast_enhanced_clipped = np.clip(contrast_enhanced, 0, 255).astype(np.uint8)

#         out = Image.fromarray(contrast_enhanced_clipped)
#         out.save(after_path)

#         return render_template('index.html', before_image='before.png', after_image='hasil.png')

#     return render_template('index.html')

from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Manual 2D convolution
def manual_convolve2d(image, kernel):
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(image_h):
        for j in range(image_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output

@app.route('/', methods=['GET', 'POST'])
def index():
    before_path = None
    after_path = None

    if request.method == 'POST':
        f = request.files['image']
        before_path = os.path.join(app.config['UPLOAD_FOLDER'], 'before.png')
        after_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hasil.png')

        # Load & konversi gambar ke grayscale
        image = Image.open(f).convert('L')
        image.save(before_path)

        img = np.array(image).astype(np.float32)

        # High-pass filter (penajaman)
        high_pass_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = manual_convolve2d(img, high_pass_kernel)
        sharpened_clipped = np.clip(sharpened, 0, 255).astype(np.uint8)

        # Gaussian blur ringan
        gaussian_kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]) / 16.0
        blurred_final = manual_convolve2d(sharpened_clipped, gaussian_kernel)
        blurred_final_clipped = np.clip(blurred_final, 0, 255).astype(np.uint8)

        # Peningkatan kontras sederhana
        G = 1.3
        P = np.mean(blurred_final_clipped)
        contrast_enhanced = G * (blurred_final_clipped.astype(np.float32) - P) + P
        contrast_enhanced_clipped = np.clip(contrast_enhanced, 0, 255).astype(np.uint8)

        # Simpan hasil akhir
        out = Image.fromarray(contrast_enhanced_clipped)
        out.save(after_path)

        return render_template('index.html', before_image='before.png', after_image='hasil.png')

    return render_template('index.html')

@app.route('/simulasi', methods=['GET', 'POST'])
def simulasi():
    processed = None
    if request.method == 'POST':
        method = request.form['method']
        f = request.files['image']
        image = Image.open(f).convert('L')
        img = np.array(image).astype(np.float32)
        img_norm = img / 255.0

        if method == 'lucy':
            psf = psf_kernel(5, 2.0)
            result = richardson_lucy(img_norm, psf, num_iter=10)
            result = np.clip(result * 255, 0, 255).astype(np.uint8)

        elif method == 'highpass':
            kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
            result = convolve2d(img, kernel, mode='same', boundary='symm')
            result = np.clip(result, 0, 255).astype(np.uint8)

        elif method == 'gaussian':
            kernel = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16.0
            result = convolve2d(img, kernel, mode='same', boundary='symm')
            result = np.clip(result, 0, 255).astype(np.uint8)

        elif method == 'kontras':
            G = 1.3
            P = np.mean(img)
            result = G * (img - P) + P
            result = np.clip(result, 0, 255).astype(np.uint8)

        elif method == 'dft':
            dft = np.fft.fft2(img)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
            result = np.clip(magnitude_spectrum, 0, 255).astype(np.uint8)

        elif method == 'threshold':
            result = (img > 127) * 255
            result = result.astype(np.uint8)

        elif method == 'saltpepper':
            prob = 0.05
            noisy = img.copy()
            rnd = np.random.rand(*img.shape)
            noisy[rnd < prob/2] = 0
            noisy[rnd > 1 - prob/2] = 255
            result = noisy.astype(np.uint8)

        elif method == 'sobel':
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            gx = convolve2d(img, sobel_x, mode='same', boundary='symm')
            gy = convolve2d(img, sobel_y, mode='same', boundary='symm')
            magnitude = np.sqrt(gx**2 + gy**2)
            result = np.clip(magnitude, 0, 255).astype(np.uint8)

        else:
            result = img.astype(np.uint8)

        out = Image.fromarray(result)
        out.save('static/simulasi_hasil.png')
        processed = 'simulasi_hasil.png'

    return render_template('simulasi.html', processed=processed)

if __name__ == '__main__':
    app.run(debug=True)
