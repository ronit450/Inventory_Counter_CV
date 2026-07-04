import os
import json
import glob
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import rasterio
from sklearn.decomposition import PCA
from ultralytics import YOLO

class YOLOUtils:
    def __init__(self):
        pass

    
    def reduce_bands_with_pca_and_replace(self, input_folder, n_components=3):
        """
        Reduce the bands of images in the input folder using PCA and replace the original image files.
        Assumes images are in a format readable by rasterio (e.g., GeoTIFF for multispectral satellite images).
        """
        for input_path in os.listdir(input_folder):
            full_input_path = os.path.join(input_folder, input_path)
            if not input_path.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):  # Process only TIFF files
                continue

            with rasterio.open(full_input_path) as src:
                img_data = src.read()  # img_data shape: (bands, height, width)
                img_data = np.transpose(img_data, (1, 2, 0))
                height, width, bands = img_data.shape
                flattened_img = img_data.reshape((height * width, bands))


                pca = PCA(n_components=n_components)
                img_reduced = pca.fit_transform(flattened_img)
                img_reduced_reshaped = img_reduced.reshape((height, width, n_components))
                img_normalized = ((img_reduced_reshaped - img_reduced_reshaped.min()) * (1/(img_reduced_reshaped.max() - img_reduced_reshaped.min()) * 255)).astype('uint8')


                img_output = Image.fromarray(img_normalized)
                new_path = os.path.splitext(full_input_path)[0] + '_RGB.png'
                img_output.save(new_path)
                # os.remove(full_input_path)
                # os.rename(new_path, full_input_path)


    def coco_to_yolo(self, input_path, output_dir):
        with open(input_path, "r") as f:
            data = json.load(f)

        images = {}
        for img in data["images"]:
            images[img["id"]] = {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            }

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for ann in data["annotations"]:
            image_info = images[ann["image_id"]]
            width = image_info["width"]
            height = image_info["height"]

            x_center = (ann["bbox"][0] + ann["bbox"][2] / 2) / width
            y_center = (ann["bbox"][1] + ann["bbox"][3] / 2) / height
            box_width = ann["bbox"][2] / width
            box_height = ann["bbox"][3] / height

            category_id = ann["category_id"] - 1  # Assuming category ids start from 1

            output_path = os.path.join(
                output_dir, os.path.splitext(image_info["file_name"])[0] + ".txt"
            )
            with open(output_path, "a") as f:
                f.write(f"{category_id} {x_center} {y_center} {box_width} {box_height}\n")
    
    

    def labelme_to_yolo(self, labelme_folder, output_folder, class_names):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for json_file_path in glob.glob(os.path.join(labelme_folder, "*.json")):
            with open(json_file_path, "r") as f:
                data = json.load(f)

                img_width = data["imageWidth"]
                img_height = data["imageHeight"]

                yolo_data = []

                for shape in data["shapes"]:
                    points = shape["points"]
                    label = shape["label"]
                    if not points:
                        continue

                    x_min = min([point[0] for point in points])
                    y_min = min([point[1] for point in points])
                    x_max = max([point[0] for point in points])
                    y_max = max([point[1] for point in points])

                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    class_idx = class_names.index(label)
                    yolo_data.append(f"{class_idx} {x_center} {y_center} {width} {height}")

                output_txt_path = os.path.join(
                    output_folder,
                    os.path.splitext(os.path.basename(json_file_path))[0] + ".txt",
                )
                with open(output_txt_path, "w") as f:
                    f.write("\n".join(yolo_data))
    def new_format_to_yolo(self, label_folder, output_folder, class_names):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for json_file_path in glob.glob(os.path.join(label_folder, "*.json")):
            with open(json_file_path, "r") as f:
                data = json.load(f)

                img_width = data["ImageWidth"]
                img_height = data["ImageHeight"]

                yolo_data = []

                for detection in data.get("detections", []):
                    box = detection["box"]
                    label = detection["name"]
                    
                    x1 = box["x1"]
                    y1 = box["y1"]
                    x2 = box["x2"]
                    y2 = box["y2"]

                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    class_idx = class_names.index(label)
                    yolo_data.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                output_txt_path = os.path.join(
                    output_folder,
                    os.path.splitext(os.path.basename(json_file_path))[0] + ".txt",
                )
                with open(output_txt_path, "w") as f:
                    f.write("\n".join(yolo_data))

    def labelme_to_yolo_segmentation(self, labelme_folder, output_folder, class_names):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for json_file_path in glob.glob(os.path.join(labelme_folder, "*.json")):
            with open(json_file_path, "r") as file:
                data = json.load(file)
                img_width = data.get("imageWidth", 1)
                img_height = data.get("imageHeight", 1)
                segmentation_data = []

                for shape in data["shapes"]:
                    label = shape["label"]
                    points = shape["points"]
                    if not points or len(points) < 2:
                        continue  # Ensure there are enough points to form a rectangle

                    # Assuming points are given as top-left and bottom-right
                    x_min = points[0][0]
                    y_min = points[0][1]
                    x_max = points[1][0]
                    y_max = points[1][1]

                    # Convert corners into normalized format
                    points_normalized = [
                        [x_min / img_width, y_min / img_height],  # Top-left
                        [x_max / img_width, y_min / img_height],  # Top-right
                        [x_max / img_width, y_max / img_height],  # Bottom-right
                        [x_min / img_width, y_max / img_height]   # Bottom-left
                    ]

                    # Flatten list and format as string
                    points_str = ' '.join(f"{x:.6f} {y:.6f}" for point in points_normalized for x, y in [point])

                    class_idx = class_names.index(label)
                    segmentation_data.append(f"{class_idx} {points_str}")

                output_txt_path = os.path.join(
                    output_folder,
                    os.path.splitext(os.path.basename(json_file_path))[0] + ".txt",
                )
                with open(output_txt_path, "w") as file:
                    file.write("\n".join(segmentation_data))


    def yolo_predict(self, model_file, img_file, out_geojson):
        model = YOLO(model_file)
        results = model(img_file)
        labels = results[0].tojson()
        self.file_to_geojson(labels, out_geojson)
        print("Detected..")

    def jpg_to_numpy(self, labelme_folder, output_folder,):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for jpeg_file_path in glob.glob(os.path.join(labelme_folder, "*.jpg")): 
            with rasterio.open(jpeg_file_path) as src:
                multi_band_image = src.read()
            output_numpy_path = os.path.join(
                    output_folder,
                    os.path.splitext(os.path.basename(jpeg_file_path))[0] + ".npy",
                )
            np.save(output_numpy_path, multi_band_image)
    
    def labelme_to_polygon_segmentation(self, labelme_folder, output_folder, class_names):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for json_file_path in glob.glob(os.path.join(labelme_folder, "*.json")):
            with open(json_file_path, "r") as file:
                data = json.load(file)
                img_width = data.get("imageWidth", 1)
                img_height = data.get("imageHeight", 1)
                segmentation_data = []

                for shape in data["shapes"]:
                    label = shape["label"]
                    points = shape["points"]
                    if not points:
                        continue

                    # Normalize polygon coordinates
                    points_normalized = ' '.join(f"{x/img_width:.6f} {y/img_height:.6f}" for point in points for x, y in [point])

                    class_idx = class_names.index(label)
                    segmentation_data.append(f"{class_idx} {points_normalized}")

                output_txt_path = os.path.join(
                    output_folder,
                    os.path.splitext(os.path.basename(json_file_path))[0] + ".txt",
                )
                with open(output_txt_path, "w") as file:
                    file.write("\n".join(segmentation_data))




    def plot_geojson(self, input_file, img_file):
        img = Image.open(img_file)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, img.width)
        ax.set_ylim(img.height, 0)
        ax.imshow(img)

        gdf = gpd.read_file(input_file)
        for count, geometry in enumerate(gdf.geometry):
            if geometry.geom_type == "Polygon":
                x, y = geometry.exterior.xy
                ax.plot(x, y)
                centroid = geometry.centroid
                label = f"{gdf['class'].values[count]} ({gdf['confidence'].values[count]:.2f})"
                ax.text(centroid.x, centroid.y, label, color="black", fontsize=8, ha="center")
        plt.show()

   


    def labelme_to_polygon_segmentation(self, labelme_folder, output_folder, class_names):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for json_file_path in glob.glob(os.path.join(labelme_folder, "*.json")):
            with open(json_file_path, "r") as file:
                data = json.load(file)
                img_width = data.get("imageWidth", 1)
                img_height = data.get("imageHeight", 1)
                segmentation_data = []

                for shape in data["shapes"]:
                    label = shape["label"]
                    points = shape["points"]
                    if not points:
                        continue

                    # Normalize polygon coordinates
                    points_normalized = ' '.join(f"{x/img_width:.6f} {y/img_height:.6f}" for point in points for x, y in [point])

                    class_idx = class_names.index(label)
                    segmentation_data.append(f"{class_idx} {points_normalized}")

                output_txt_path = os.path.join(
                    output_folder,
                    os.path.splitext(os.path.basename(json_file_path))[0] + ".txt",
                )
                with open(output_txt_path, "w") as file:
                    file.write("\n".join(segmentation_data))


test_folder = r"C:\Users\User\Downloads\tomatoe_dataset\ds0\img"
output_folder = r"C:\Users\User\Downloads\tomatoe_dataset\ds0\img"
class_names = ["tomato"]
utils = YOLOUtils()
# utils.reduce_bands_with_pca_and_replace(test_folder)
utils.labelme_to_yolo(test_folder, output_folder, class_names)
# utils.jpg_to_numpy(test_folder, output_folder)
    