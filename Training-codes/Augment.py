import os
import shutil
import json
import cv2
import glob
from PIL import Image


class Augment:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.augmentations = ["horizontal_flip", "vertical_flip", "rotate_180"]
        self.augmented_dirs = {aug: os.path.join(input_dir, aug) for aug in self.augmentations}
        self.image_extensions = (".JPG", ".JPEG",".jpeg", ".png", ".tif", ".tiff", '.jpg')
        self._ensure_output_dirs()
    
    def _ensure_output_dirs(self):
        for directory in self.augmented_dirs.values():
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def get_image_files(self):
        return [f for f in os.listdir(self.input_dir) if f.lower().endswith(self.image_extensions)]
    
    def process_images(self):
        self.horizontal_flip()
        self.vertical_flip()
        self.rotate_180()
        self.merge_and_cleanup()
    
    def horizontal_flip(self):
        self._apply_transformation(cv2.flip, 1, self.augmented_dirs["horizontal_flip"], "new_h_flip")
    
    def vertical_flip(self):
        self._apply_transformation(cv2.flip, 0, self.augmented_dirs["vertical_flip"], "new_v_flip")
    
    def rotate_180(self):
        self._apply_transformation(self._rotate_180, None, self.augmented_dirs["rotate_180"], "new_r_180")
    
    def _apply_transformation(self, transform, param, output_dir, suffix):
        all_images = self.get_image_files()
        for filename in all_images:
            base_name, file_ext = os.path.splitext(filename)
            image_path = os.path.join(self.input_dir, filename)
            json_path = os.path.join(self.input_dir, base_name + ".json")
            output_image_path = os.path.join(output_dir, f"{base_name}_{suffix}{file_ext}")
            output_json_path = os.path.join(output_dir, f"{base_name}_{suffix}.json")
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            transformed_image = transform(image, param) if param is not None else transform(image)
            cv2.imwrite(output_image_path, transformed_image)
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                    data['imagePath'] = os.path.basename(output_image_path)
                    for shape in data['shapes']:
                        for point in shape['points']:
                            if param == 1:
                                point[0] = image.shape[1] - point[0] - 1
                            elif param == 0:
                                point[1] = image.shape[0] - point[1] - 1
                            else:
                                point[0] = image.shape[1] - point[0] - 1
                                point[1] = image.shape[0] - point[1] - 1
                    with open(output_json_path, 'w') as file:
                        json.dump(data, file, indent=4)
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError in {json_path}: {e}")

            txt_path = os.path.join(self.input_dir, base_name + ".txt")
            output_txt_path = os.path.join(output_dir, f"{base_name}_{suffix}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as file:
                    lines = file.readlines()
                transformed_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = parts[0]
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        if param == 1:
                            x_center = 1.0 - x_center
                        elif param == 0:
                            y_center = 1.0 - y_center
                        else:
                            x_center = 1.0 - x_center
                            y_center = 1.0 - y_center
                        transformed_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                with open(output_txt_path, 'w') as file:
                    file.writelines(transformed_lines)
        
    def _rotate_180(self, image):
        return cv2.rotate(image, cv2.ROTATE_180)
    
    def merge_and_cleanup(self):
        for aug_dir in self.augmented_dirs.values():
            for file in os.listdir(aug_dir):
                shutil.move(os.path.join(aug_dir, file), os.path.join(self.input_dir, file))
            shutil.rmtree(aug_dir)


if __name__ == "__main__":
    input_directory = r"C:\Users\User\Downloads\maryam_cleaned_it\bbox_training\only_corty"
    augmentor = Augment(input_directory)
    augmentor.process_images()
