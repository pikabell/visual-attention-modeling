# Computational Modeling of Visual Attention

This project implements a computational model of visual attention based on the work by Itti & Koch (1998). The model simulates human visual attention mechanisms by generating saliency maps and use them for image cropping and classification.

## Model Architecture

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 200">
  <!-- Input Image Box -->
  <rect x="50" y="50" width="100" height="60" fill="none" stroke="black"/>
  <text x="100" y="85" text-anchor="middle" font-size="12">Input Image</text>
  
  <!-- Feature Maps -->
  <rect x="200" y="20" width="80" height="40" fill="none" stroke="black"/>
  <text x="240" y="45" text-anchor="middle" font-size="10">Color Maps</text>
  
  <rect x="200" y="70" width="80" height="40" fill="none" stroke="black"/>
  <text x="240" y="95" text-anchor="middle" font-size="10">Intensity Maps</text>
  
  <rect x="200" y="120" width="80" height="40" fill="none" stroke="black"/>
  <text x="240" y="145" text-anchor="middle" font-size="10">Orientation Maps</text>
  
  <!-- Conspicuity Maps -->
  <rect x="350" y="45" width="100" height="90" fill="none" stroke="black"/>
  <text x="400" y="90" text-anchor="middle" font-size="12">Conspicuity Maps</text>
  
  <!-- Saliency Map -->
  <rect x="500" y="60" width="90" height="60" fill="none" stroke="black"/>
  <text x="545" y="90" text-anchor="middle" font-size="12">Saliency Map</text>
  
  <!-- Classification -->
  <rect x="650" y="60" width="90" height="60" fill="none" stroke="black"/>
  <text x="695" y="90" text-anchor="middle" font-size="12">ResNet</text>
  
  <!-- Arrows -->
  <path d="M150 80 L200 40" stroke="black" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M150 80 L200 90" stroke="black" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M150 80 L200 140" stroke="black" fill="none" marker-end="url(#arrowhead)"/>
  
  <path d="M280 40 L350 90" stroke="black" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M280 90 L350 90" stroke="black" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M280 140 L350 90" stroke="black" fill="none" marker-end="url(#arrowhead)"/>
  
  <path d="M450 90 L500 90" stroke="black" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M590 90 L650 90" stroke="black" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Arrow Marker -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="black"/>
    </marker>
  </defs>
</svg>

## Features

- Implementation of Itti & Koch's visual attention model
- Generation of feature maps (color, intensity, orientation)
- Conspicuity maps computation
- Saliency map generation
- Image cropping based on saliency map
- Image classification using ResNet

