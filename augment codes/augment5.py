import os
import cv2
import albumentations as A
from random import choice

def augment_images(image_paths, augmenter, num_augmented_images):
    augmented_images = []
    for _ in range(num_augmented_images):
        image_path = choice(image_paths)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = augmenter(image=image)
        augmented_image = augmented['image']
        augmented_images.append((augmented_image, image_path))
    return augmented_images

def save_images(images, output_folder, base_name, source_images_to_save):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, (image, source_path) in enumerate(images):
        output_image_path = os.path.join(output_folder, f"{base_name}_aug_{i}.png")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, image)

    for i, source_path in enumerate(source_images_to_save):
        output_source_path = os.path.join(output_folder, f"{base_name}_source_{i}.png")
        source_image = cv2.imread(source_path)
        cv2.imwrite(output_source_path, source_image)

def dynamic_balancing_augmentation(input_folder_class1, input_folder_class2, output_folder_class1, output_folder_class2, augmenter, target_count):
    class1_images = [os.path.join(input_folder_class1, f) for f in os.listdir(input_folder_class1) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    class2_images = [os.path.join(input_folder_class2, f) for f in os.listdir(input_folder_class2) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    num_class1 = len(class1_images)
    num_class2 = len(class2_images)

    if num_class1 < target_count:
        num_augmentations_class1 = (target_count+num_class1) - num_class1
        augmented_class1 = augment_images(class1_images, augmenter, num_augmentations_class1)
        source_images_class1 = class1_images[:2]  # Taking the first 2 source images to save
        save_images(augmented_class1, output_folder_class1, "class1", source_images_class1)

    if num_class2 < target_count:
        num_augmentations_class2 = (target_count+num_class2) - num_class2
        augmented_class2 = augment_images(class2_images, augmenter, num_augmentations_class2)
        source_images_class2 = class2_images[:2]  # Taking the first 2 source images to save
        save_images(augmented_class2, output_folder_class2, "class2", source_images_class2)

# Define augmentation pipeline using Albumentations
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),  # horizontally flip with 50% probability
    A.Rotate(limit=(-20, 20), p=0.5),  # rotate with a maximum angle of -20 to 20 degrees
    A.RandomBrightnessContrast(p=0.5),  # adjust brightness and contrast randomly
    # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # add Gaussian noise
    # A.Blur(blur_limit=(3, 7), p=0.5),  # apply blur
    # A.RGBShift(p=0.5),  # randomly shift RGB channels
])

# Define input and output folders
input_folder_class1 = 'C:/Users/Aditi Bolakhe/Desktop/drowsy detection/augment/input1'
input_folder_class2 = 'C:/Users/Aditi Bolakhe/Desktop/drowsy detection/augment/input2'
output_folder_class1 = 'C:/Users/Aditi Bolakhe/Desktop/drowsy detection/augment/output1'
output_folder_class2 = 'C:/Users/Aditi Bolakhe/Desktop/drowsy detection/augment/output2'

# Target number of images for each class
target_count = 10

# Perform dynamic balancing augmentation
dynamic_balancing_augmentation(input_folder_class1, input_folder_class2, output_folder_class1, output_folder_class2, augmenter, target_count)
