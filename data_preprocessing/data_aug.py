import cv2
import numpy as np
import random
import os

def resizeDisplayImages(image_path1, image_path2, window_name="Images", resize_factor=1):
  """
  Reads two images, resizes them, and displays them side-by-side in a single window.

  Args:
      image_path1 (str): Path to the first image file.
      image_path2 (str): Path to the second image file.
      window_name (str, optional): The name of the window to display the images. Defaults to "Images".
      resize_factor (float, optional): Factor by which to resize the images. Defaults to 0.5 (half size).
  """

  # Read images using OpenCV
  image1 = cv2.imread(image_path1)
  image2 = cv2.imread(image_path2)

  # Check if images are loaded successfully
  if image1 is None or image2 is None:
    print("Error: Could not read images!")
    return

  # Get image heights and widths
  image1_height, image1_width, channels = image1.shape
  image2_height, image2_width, channels = image2.shape

  # Calculate new dimensions based on resize factor
  new_width1 = int(image1_width * resize_factor)
  new_height1 = int(image1_height * resize_factor)
  new_width2 = int(image2_width * resize_factor)
  new_height2 = int(image2_height * resize_factor)

  # Resize the images
  resized_image1 = cv2.resize(image1, (new_width1, new_height1), interpolation=cv2.INTER_AREA)
  resized_image2 = cv2.resize(image2, (new_width2, new_height2), interpolation=cv2.INTER_AREA)

  # Combine images horizontally
  combined_image = cv2.hconcat([resized_image1, resized_image2])

  # Display the combined image in a window
  cv2.imshow(window_name, combined_image)

  # Wait for a key press to close the window
  cv2.waitKey(0)

  # Close all windows
  cv2.destroyAllWindows()

# # Example usage
# texture_path = "D://psg//sem 8//Deep Learning//gan_terrain_generation//data//raw//texture//world.200411.3x21600x10800.png"
# heightmap_path = "D://psg//sem 8//Deep Learning//gan_terrain_generation//data//raw///heightmap//gebco_08_rev_elev_21600x10800.png"

# resizeDisplayImages(texture_path, heightmap_path)


def crop_and_store_patches(image_path1, image_path2, patch_size=(256, 256), stride=(256, 256)):
  """
  Crops a texture image and corresponding heightmap into overlapping patches and stores them in separate lists.

  Args:
      image_path1 (str): Path to the texture image file.
      image_path2 (str): Path to the heightmap image file.
      patch_size (tuple, optional): The size of each image patch. Defaults to (256, 256).
      stride (tuple, optional): The stride for moving the cropping window. Defaults to (256, 256). 

  Returns:
      tuple: A tuple containing two lists:
          - List of preprocessed texture image patches (NumPy arrays).
          - List of preprocessed heightmap image patches (NumPy arrays).
  """

  # Read images using OpenCV
  texture = cv2.imread(image_path1)
  heightmap = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)  # Read heightmap as grayscale

  # Get image dimensions
  texture_height, texture_width, channels = texture.shape
  heightmap_height, heightmap_width = heightmap.shape

  # Ensure patch size doesn't exceed image size
  patch_size = min(patch_size[0], texture_width), min(patch_size[1], texture_height)

  # Calculate number of patches (with potential remainder)
  num_patches_x = (texture_width - patch_size[0]) // stride[0] + 1
  num_patches_y = (texture_height - patch_size[1]) // stride[1] + 1

  # Initialize empty lists to store patches
  texture_patches = []
  heightmap_patches = []

  # Loop through image with specified stride to create patches
  for y in range(0, texture_height, stride[1]):
    for x in range(0, texture_width, stride[0]):
      # Extract patch coordinates (ensure they stay within image bounds)
      end_y = min(y + patch_size[1], texture_height)
      end_x = min(x + patch_size[0], texture_width)

      # Crop texture and heightmap patches
      texture_patch = texture[y:end_y, x:end_x]
      heightmap_patch = heightmap[y:end_y, x:end_x]

      # Normalize pixel values (0 to 1)
      texture_patch = texture_patch.astype(np.float32) / 255.0
      heightmap_patch = heightmap_patch.astype(np.float32) / 255.0

      # Append preprocessed patches to lists
      texture_patches.append(texture_patch)
      heightmap_patches.append(heightmap_patch)

  # Return a tuple containing the lists of preprocessed texture and heightmap patches
  return texture_patches, heightmap_patches

def filter_patches(texture_patches, heightmap_patches, ocean_threshold=0.01, min_size=(256, 256)):
  """
  Removes texture and heightmap patch pairs where the heightmap indicates mostly ocean and the patch size is below a minimum threshold.

  Args:
      texture_patches (list): A list containing the texture image patches as NumPy arrays.
      heightmap_patches (list): A list containing the heightmap image patches as NumPy arrays.
      ocean_threshold (float, optional): The threshold value for average heightmap pixel value to consider a patch as ocean. Defaults to 0.1.
      min_size (tuple, optional): The minimum size requirement for the patches (width, height). Defaults to (256, 256).

  Returns:
      tuple: A tuple containing the filtered texture and heightmap patch lists (without ocean patches or patches below minimum size).
  """

  filtered_texture_patches = []
  filtered_heightmap_patches = []

  for texture_patch, heightmap_patch in zip(texture_patches, heightmap_patches):
    # Calculate patch height, width, and channels
    patch_height, patch_width, channels = texture_patch.shape

    # Check if patch size meets minimum requirement
    if patch_width < min_size[0] or patch_height < min_size[1]:
      continue  # Skip patch pair (below minimum size)

    # Calculate average pixel value of the heightmap patch
    average_height = np.mean(heightmap_patch)

    # Check if average height is below the ocean threshold
    if average_height < ocean_threshold:
      continue  # Skip patch pair (considered ocean)

    # Add patch pair to filtered lists if not ocean and meets size requirement
    filtered_texture_patches.append(texture_patch)
    filtered_heightmap_patches.append(heightmap_patch)

  return filtered_texture_patches, filtered_heightmap_patches

def shuffle_lists(filtered_texture_patches, filtered_heightmap_patches):
    # Combine the lists using zip
    combined_lists = list(zip(filtered_texture_patches, filtered_heightmap_patches))

    # Shuffle the combined list
    random.shuffle(combined_lists)

    # Unzip the shuffled list
    shuffled_texture_patches, shuffled_heightmap_patches = zip(*combined_lists)

    return list(shuffled_texture_patches), list(shuffled_heightmap_patches)


def save_patches(patches, save_dir, prefix="patch_", starting_index=0):
  """
  Saves a list of image patches as individual PNG files in a specified directory.

  Args:
      patches (list): A list containing the image patches as NumPy arrays.
      save_dir (str): The path to the directory where the files will be saved.
      prefix (str, optional): A prefix to add to the filenames. Defaults to "patch_".
      starting_index (int, optional): The starting index for filename numbering. Defaults to 0.
  """

  # Ensure the save directory exists
  os.makedirs(save_dir, exist_ok=True)

  for i, patch in enumerate(patches):
    # Generate filename with prefix and zero-padded index
    filename = f"{prefix}{starting_index + i:04d}.png"
    save_path = os.path.join(save_dir, filename)

    # Check if patch has alpha channel (assuming last dimension is channels)
    if len(patch.shape) == 3 and patch.shape[2] == 4:
      # Save with alpha channel preserved (assuming RGBA format)
      cv2.imwrite(save_path, patch, flags=-1)
    else:
      # Rescale if necessary (assuming normalization between 0 and 1)
      rescaled_patch = patch * 255  # Adjust scaling factor if needed based on normalization
      cv2.imwrite(save_path, rescaled_patch.astype(np.uint8))  # Convert to uint8 for image data

    



texture_path = "D://psg//sem 8//Deep Learning//gan_terrain_generation//data//raw//texture//world.200411.3x21600x10800.png"
heightmap_path = "D://psg//sem 8//Deep Learning//gan_terrain_generation//data//raw///heightmap//gebco_08_rev_elev_21600x10800.png"

patch_size = (256, 256)  # Adjust patch size as needed
stride = (256, 256)  # Adjust stride as needed (consider overlapping patches if desired)

texture_patches, heightmap_patches = crop_and_store_patches(texture_path, heightmap_path, patch_size, stride)

# Now you have two separate lists containing preprocessed texture and heightmap patches
print(f"Number of texture patches: {len(texture_patches)}")
print(f"Number of heightmap patches: {len(heightmap_patches)}")

# cv2.imshow("First Texture Patch", first_texture_patch)
# cv2.waitKey(0)  # Wait for a key press to close the window
# cv2.destroyAllWindows()


ocean_threshold = 0.01  # Adjust this value as needed based on your heightmap data

filtered_texture_patches, filtered_heightmap_patches = filter_patches(texture_patches, heightmap_patches, ocean_threshold,(256,256))

# Filter out oceans and patches below 256x256
print(f"Number of filtered texture patches: {len(filtered_texture_patches)}")
print(f"Number of filtered heightmap patches: {len(filtered_heightmap_patches)}")

shuffled_texture_patches, shuffled_heightmap_patches = shuffle_lists(filtered_texture_patches, filtered_heightmap_patches)

# for i in range(570,590):
#   cv2.imshow("First Texture Patch", shuffled_texture_patches[i])
#   cv2.waitKey(0)  # Wait for a key press to close the window
#   cv2.destroyAllWindows()


save_dir_texture = "D://psg//sem 8//Deep Learning//gan_terrain_generation//data//preprocessed//texture"  
save_dir_heightmap = "D://psg//sem 8//Deep Learning//gan_terrain_generation//data//preprocessed//heightmap"  

# Save texture patches
save_patches(shuffled_texture_patches, save_dir_texture, prefix="texture_", starting_index=100)

# Save heightmap patches (assuming same order as texture)
save_patches(shuffled_heightmap_patches, save_dir_heightmap, prefix="heightmap_", starting_index=100)

print("Patches saved successfully!")


# ------------------------------------------------------------
# Add normalization before training.
# texture_patch = texture_patch.astype(np.float32) / 255.0
# heightmap_patch = heightmap_patch.astype(np.float32) / 255.0
