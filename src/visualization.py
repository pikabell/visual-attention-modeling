import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class SaliencyVisualizer:
    def __init__(self, image, saliency_map):
        self.image = image
        self.saliency_map = saliency_map
    
    def overlay_saliency(self, threshold=0.5, alpha=0.5):
        saliency_resized = cv2.resize(self.saliency_map, 
                                    (self.image.shape[1], self.image.shape[0]))
        _, binary_mask = cv2.threshold(saliency_resized, threshold, 1, 
                                     cv2.THRESH_BINARY)
        binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)
        heatmap = cv2.applyColorMap((binary_mask * 255).astype(np.uint8), 
                                   cv2.COLORMAP_INFERNO)
        
        if len(self.image.shape) == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return cv2.addWeighted(self.image, 1 - alpha, heatmap, alpha, 0)
    
    def plot_3d_visualization(self):
        x = np.arange(self.saliency_map.shape[1])
        y = np.arange(self.saliency_map.shape[0])
        x, y = np.meshgrid(x, y)
        
        fig = go.Figure()
        
        image_resized = cv2.resize(self.image, 
                                 (self.saliency_map.shape[1], 
                                  self.saliency_map.shape[0]))
        image_normalized = image_resized / 255.0
        
        fig.add_trace(go.Surface(
            z=np.zeros_like(self.saliency_map),
            x=x, y=y,
            surfacecolor=image_normalized[..., 0],
            colorscale='gray',
            showscale=False,
            name="Background Image"
        ))
        
        fig.add_trace(go.Surface(
            z=self.saliency_map,
            x=x, y=y,
            surfacecolor=self.saliency_map,
            colorscale='Viridis',
            opacity=0.4,
            showscale=True,
            name="Saliency Map"
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X-axis", visible=True),
                yaxis=dict(title="Y-axis", visible=True),
                zaxis=dict(title="Saliency Value", visible=True),
                aspectratio=dict(x=1, y=1, z=0.5),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            title="Interactive 3D Saliency Map",
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
