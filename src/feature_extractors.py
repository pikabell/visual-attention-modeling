import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape[:2]
    
    def get_intensity(self):
        return cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
    
    def get_color_channels(self):
        R, G, B = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]
        total = R + G + B + 1e-6
        return R/total, G/total, B/total
    
    def get_color_opponency(self):
        R, G, B = self.get_color_channels()
        RG = np.abs(R - G)
        BY = np.abs(B - (R + G) / 2)
        return RG, BY
    
    def get_orientation(self, angles=[0, 45, 90, 135]):
        intensity = self.get_intensity()
        orientations = []
        for angle in angles:
            theta = np.deg2rad(angle)
            kernel = cv2.getGaborKernel((31, 31), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(intensity, cv2.CV_32F, kernel)
            orientations.append(filtered)
        return orientations
