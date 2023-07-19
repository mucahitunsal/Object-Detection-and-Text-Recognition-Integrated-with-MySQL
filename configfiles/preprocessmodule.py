import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self):
        pass

    # Frame'i yeniden boyutlandırma
    def rescale(self, frame, scale_percent=0.8):
        width = int(frame.shape[1] * scale_percent)
        height = int(frame.shape[0] * scale_percent)
        dim = (width, height)

        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Frame'e gri filtre uygulama
    def gray_filter(self, frame):

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Frame'e thresholding işlemi uygulamak

    def thresholding(self, frame, threshold_value=190):
        _, thresh = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)

        return thresh

    # Frame üzerindeki gürültüyü azaltmak
    def remove_noise(self, frame):

        return cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 1)

    # Frame üzerindeki yazıları inceltmek


    def thin_font(self, frame):
        kernel = np.ones((1, 1), np.uint8)
        # iterations değeri db'den ürün bazlı okunabilir
        return cv2.erode(frame, kernel, iterations=1)


    # Frame üzerindeki yazıları kalınlaştırmak

    def thick_font(self, frame):
        kernel = np.ones((1, 1), np.uint8)

        return cv2.dilate(frame, kernel, iterations=1)

    # Frame'in fazlalık kısımlarını atmak

    def remove_borders(self, frame):
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        cnt = contours_sorted[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = frame[y:y+h, x:x+w]

        return crop

    # Frame'in perspektifini düzeltmek

    def correct_perspective(self, frame):

        corner_detection_params = dict(
            blockSize=2,
            ksize=3,
            k=0.04
        )

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners = cv2.cornerHarris(gray_frame, **corner_detection_params)

        corners = cv2.dilate(corners, None)

        threshold = 0.01 * corners.max()
        corners_indices = np.where(corners > threshold)
        corners = np.float32(np.column_stack((corners_indices[1], corners_indices[0])))

        src_points = corners
        dst_points = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        corrected_image = cv2.warpPerspective(frame, perspective_matrix, (800, 800))

        return corrected_image

