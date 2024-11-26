import cv2
import numpy as np

class GaussianPyramid:
    def __init__(self, image, levels=5):
        self.image = image
        self.levels = levels
        self.pyramid = self._build_pyramid()
    
    def _build_pyramid(self):
        pyramid = [self.image]
        for x in range(1, self.levels):
            blurred = cv2.GaussianBlur(pyramid[-1], (5, 5), 1.0)
            downsampled = cv2.pyrDown(blurred)
            pyramid.append(downsampled)
        return pyramid
    
    def get_center_surround_differences(self, level_pairs=[(2, 4), (3, 5)]):
        differences = []
        for center, surround in level_pairs:
            if center >= len(self.pyramid) or surround >= len(self.pyramid):
                continue
            upsampled = cv2.pyrUp(self.pyramid[surround])
            if upsampled.shape != self.pyramid[center].shape:
                upsampled = cv2.resize(upsampled, 
                                     (self.pyramid[center].shape[1], 
                                      self.pyramid[center].shape[0]))
            diff = cv2.absdiff(self.pyramid[center], upsampled)
            differences.append(diff)
        return differences