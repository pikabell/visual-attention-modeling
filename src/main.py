# import os
# import cv2
# import matplotlib.pyplot as plt
# from saliency_map import SaliencyMap
# from visualization import SaliencyVisualizer

# def main():
#     image_path = "../images/img2.jpg"  # The path to image
#     image_name = os.path.splitext(os.path.basename(image_path))[0]  
    
#     results_dir = f"results_{image_name}"
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)

#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (400, 300))

#     saliency_detector = SaliencyMap(image)
#     saliency_map, intensity, rg, by, orientation = saliency_detector.compute_saliency()

#     visualizer = SaliencyVisualizer(image, saliency_map)
#     overlay = visualizer.overlay_saliency()

#     overlay_image_path = os.path.join(results_dir, "saliency_overlay.jpg")
#     cv2.imwrite(overlay_image_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
#     print(f"Saliency overlay saved at: {overlay_image_path}")

#     intensity_image_path = os.path.join(results_dir, "intensity.png")
#     cv2.imwrite(intensity_image_path, intensity) 
#     print(f"Intensity image saved at: {intensity_image_path}")

#     rg_image_path = os.path.join(results_dir, "RG.png")
#     plt.imsave(rg_image_path, rg,cmap='hot') 
#     print(f"RG image saved at: {rg_image_path}")

#     by_image_path = os.path.join(results_dir, "BY.png")
#     plt.imsave(by_image_path, by,cmap='hot')  
#     print(f"BY image saved at: {by_image_path}")

#     for i, orientation_image in enumerate(orientation):
#         print(orientation_image.shape)
#         orientation_image_path = os.path.join(results_dir, f"orientation_{i+1}.png")
#         plt.imsave(orientation_image_path, orientation_image,cmap='gray')  
#         print(f"Orientation image {i+1} saved at: {orientation_image_path}")

#     fig = visualizer.plot_3d_visualization()
#     plot_html_path = os.path.join(results_dir, "saliency_3d_visualization.html")
#     fig.write_html(plot_html_path)
#     print(f"3D visualization saved at: {plot_html_path}")

# if __name__ == "__main__":
#     main()
import os
import cv2
import glob
import matplotlib.pyplot as plt
from saliency_map import SaliencyMap
from visualization import SaliencyVisualizer

def main():
    # Folder containing your images
    images_folder = "../images"  # Path to the images folder
    
    # Get all image files in the folder (you can modify the pattern to match other formats as needed)
    image_paths = glob.glob(os.path.join(images_folder, "*.jpg"))  # You can adjust the pattern for other image types like *.png
    
    # Create a top-level "results" directory if it doesn't exist
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Process each image in the folder
    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract base name without extension
        
        # Create a subdirectory for each image inside the results directory
        image_results_dir = os.path.join(results_dir, image_name)
        if not os.path.exists(image_results_dir):
            os.makedirs(image_results_dir)

        # Read and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (400, 300))  # Resize if needed

        # Compute saliency maps and additional outputs
        saliency_detector = SaliencyMap(image)
        saliency_map, intensity, rg, by, orientation = saliency_detector.compute_saliency()

        # Visualize and save saliency overlay
        visualizer = SaliencyVisualizer(image, saliency_map)
        overlay = visualizer.overlay_saliency()

        overlay_image_path = os.path.join(image_results_dir, "saliency_overlay.jpg")
        cv2.imwrite(overlay_image_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))  # Save as BGR for OpenCV
        print(f"Saliency overlay saved at: {overlay_image_path}")

        # Save intensity image
        intensity_image_path = os.path.join(image_results_dir, "intensity.png")
        cv2.imwrite(intensity_image_path, intensity)
        print(f"Intensity image saved at: {intensity_image_path}")

        # Save RG image using matplotlib (with 'hot' colormap)
        rg_image_path = os.path.join(image_results_dir, "RG.png")
        plt.imsave(rg_image_path, rg, cmap='hot')
        print(f"RG image saved at: {rg_image_path}")

        # Save BY image using matplotlib (with 'hot' colormap)
        by_image_path = os.path.join(image_results_dir, "BY.png")
        plt.imsave(by_image_path, by, cmap='hot')
        print(f"BY image saved at: {by_image_path}")

        # Save orientation images (one per file)
        for i, orientation_image in enumerate(orientation):
            orientation_image_path = os.path.join(image_results_dir, f"orientation_{i+1}.png")
            plt.imsave(orientation_image_path, orientation_image, cmap='gray')  # Save with gray colormap
            print(f"Orientation image {i+1} saved at: {orientation_image_path}")

        # Create 3D visualization and save it
        fig = visualizer.plot_3d_visualization()
        plot_html_path = os.path.join(image_results_dir, "saliency_3d_visualization.html")
        fig.write_html(plot_html_path)
        print(f"3D visualization saved at: {plot_html_path}")

if __name__ == "__main__":
    main()
