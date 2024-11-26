import cv2
import numpy as np
from feature_extractors import FeatureExtractor
from pyramid import GaussianPyramid
class SaliencyMap:
    def __init__(self, image):
        self.image = image
        self.feature_extractor = FeatureExtractor(image)
        self.saliency_map = None
        
    def compute_saliency(self):
        intensity = self.feature_extractor.get_intensity()
        RG, BY = self.feature_extractor.get_color_opponency()
        orientations = self.feature_extractor.get_orientation()
        
        intensity_pyramid = GaussianPyramid(intensity)
        rg_pyramid = GaussianPyramid(RG)
        by_pyramid = GaussianPyramid(BY)
        orientation_pyramids = [GaussianPyramid(o) for o in orientations]
        
        intensity_diff = intensity_pyramid.get_center_surround_differences()
        rg_diff = rg_pyramid.get_center_surround_differences()
        by_diff = by_pyramid.get_center_surround_differences()
        orientation_diffs = [p.get_center_surround_differences() 
                           for p in orientation_pyramids]
        
        intensity_conspicuity = np.sum(intensity_diff, axis=0)
        rg_conspicuity = np.sum(rg_diff, axis=0)
        by_conspicuity = np.sum(by_diff, axis=0)
        orientation_conspicuity = [np.sum(o, axis=0) for o in orientation_diffs]
        
        normalized_intensity = cv2.normalize(intensity_conspicuity, None, 0, 1, 
                                          cv2.NORM_MINMAX)
        normalized_rg = cv2.normalize(rg_conspicuity, None, 0, 1, cv2.NORM_MINMAX)
        normalized_by = cv2.normalize(by_conspicuity, None, 0, 1, cv2.NORM_MINMAX)
        normalized_orientation = [cv2.normalize(o, None, 0, 1, cv2.NORM_MINMAX) 
                                for o in orientation_conspicuity]
        
        all_conspicuities = ([normalized_intensity, normalized_rg, normalized_by] + 
                           normalized_orientation)
        weights = [1.0, 0.7, 0.7] + [0.5] * len(normalized_orientation)
        
        self.saliency_map = self._combine_conspicuity_maps(all_conspicuities, weights)
        return self.saliency_map , intensity, RG, BY, orientations
    
    def _combine_conspicuity_maps(self, maps, weights):
        weighted_maps = [m * w for m, w in zip(maps, weights)]
        saliency = np.sum(weighted_maps, axis=0)
        return cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
