import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
import cv2
import csv
import pandas as pd

#****************************************************
# STL convert to PNG
#****************************************************
# Function to convert height to greyscale value
def height_to_greyscale(height):
    if height <= -20:
        return 0
    if height >= 0:
        return 255
    return int(255 + (height / 20) * 255)

# Function to sample multiple points within a triangle
def sample_points_in_triangle(v0, v1, v2, num_samples=20):
    samples = []
    for _ in range(num_samples):
        # Generate random barycentric coordinates
        r1 = np.random.rand()
        r2 = np.random.rand()
        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = r2 * sqrt_r1
        w = 1 - u - v
        # Calculate the sample point
        point = u * v0 + v * v1 + w * v2
        samples.append(point)
    return samples

def fill_black_pixels(image):
    def fill_pixel(values):
        # Check if the center pixel is black (value 0)
        if values[4] == 0:
            # Count the number of non-zero surrounding pixels
            non_zero_count = np.count_nonzero(values[np.arange(9) != 4])
            # If 6, 7, or 8 of the surrounding pixels are non-zero, fill the pixel
            if non_zero_count >= 7:
                non_zero_values = values[values > 0]
                return int(np.mean(non_zero_values))
        return values[4]  # Return the original value if conditions aren't met

    # Apply the filter to each pixel in the image
    filled_image = generic_filter(image, fill_pixel, size=3, mode='constant', cval=0)
    return filled_image

# Function to process STL file and create greyscale image
def process_stl_to_png(input_file, output_path, output_resolution):
    default_resolution = 320
    scaling_factor = output_resolution / default_resolution
    mesh = trimesh.load_mesh(input_file)

    # Get the bounding box of the mesh
    min_bound, max_bound = mesh.bounds
    object_width_pixels = max_bound[0] - min_bound[0]
    object_depth_pixels = max_bound[1] - min_bound[1]

    # Center offset to place the object at the image's center
    x_offset = round((output_resolution - object_width_pixels) / 2)
    y_offset = round((output_resolution - object_depth_pixels) / 2)

    # Create a 2D grid for the image, with 0 as the background
    image = np.zeros((output_resolution, output_resolution), dtype=np.uint8)

    # Translate and scale the mesh vertices
    vertices = mesh.vertices.copy()
    vertices[:, 0] = (vertices[:, 0] - min_bound[0]) + x_offset
    vertices[:, 1] = (vertices[:, 1] - min_bound[1]) + y_offset

    # Iterate over each face and sample multiple points
    for face in mesh.faces:
        v0, v1, v2 = vertices[face]
        sampled_points = sample_points_in_triangle(v0, v1, v2, num_samples=20)

        for point in sampled_points:
            x, y, z = point
            grey_value = height_to_greyscale(z)

            # Flip coordinates and ensure they are within bounds
            px = int(np.clip((output_resolution - 1 - y), 0, (output_resolution - 1)))  # Flip vertically
            py = int(np.clip(x, 0, (output_resolution - 1)))  # Flip horizontally
            image[px, py] = grey_value

    # Apply the post-processing step to fill black pixels
    image = fill_black_pixels(image)
    center_offset = 0

    # Scale the image to match output_resolution if it differs from default_resolution
    if output_resolution != default_resolution:
        new_image = np.zeros((output_resolution, output_resolution), dtype=np.uint8)
        scaled_image = np.kron(image, np.ones((int(scaling_factor), int(scaling_factor)), dtype=np.uint8))
        center_offset = (scaled_image.shape[0] - output_resolution) // 2
        new_image[:output_resolution, :output_resolution] = scaled_image[center_offset:center_offset + output_resolution, center_offset:center_offset + output_resolution]
        image = new_image

    # Scale the object dimensions
    scaled_width = round(object_width_pixels) * scaling_factor
    scaled_height = round(object_depth_pixels) * scaling_factor

    # Scale the boundaries
    scaled_min_x = (x_offset - 1) * scaling_factor - center_offset
    scaled_max_x = (round(max_bound[0] - min_bound[0]) + x_offset - 1) * scaling_factor - center_offset
    scaled_min_y = (y_offset - 1) * scaling_factor - center_offset
    scaled_max_y = (round(max_bound[1] - min_bound[1]) + y_offset - 1) * scaling_factor - center_offset

    # Generate the output file name based on the input file name
    file_name = os.path.splitext(os.path.basename(input_file))[0] + ".png"
    output_file = os.path.join(output_path, file_name)

    # Save the image
    plt.imsave(output_file, image, cmap='gray', vmin=0, vmax=255)
    print(f"Converted {input_file} to {output_file}")

    # Return the scaled dimensions and scaled boundaries
    scaled_boundaries = [scaled_min_x, scaled_max_x, scaled_min_y, scaled_max_y]
    return [scaled_width, scaled_height], image, file_name

# Main function to call the processing function
def STL_to_png(input_file, output_path, output_resolution = 320):
    object_size, converted_png, image_file_name = process_stl_to_png(input_file, output_path, output_resolution)
    return object_size, converted_png, image_file_name

#****************************************************
# Image alignment
#****************************************************
# Use the minimum frame method to locate the point image center, and move the center to the image center (320, 320)
def align_centers(points_mask):
    frame_center = calculate_object_center(points_mask)

    # Calculate the center of the object mask
    # Get the dimensions of the image
    height, width = points_mask.shape[:2]
    image_center = [width // 2, height // 2]

    # Calculate the translation needed to move the frame center to the object center
    translation = np.array(image_center) - np.array(frame_center)
    translation = [int(translation[0]), int(translation[1])]

    # Apply the translation to the points mask
    rows, cols = points_mask.shape
    M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    aligned_points_mask = cv2.warpAffine(points_mask, M, (cols, rows))

    return aligned_points_mask, translation, image_center

#Calculate the center of an object in a grayscale image.
def calculate_object_center(image):
    # Get the coordinates of all non-zero pixels
    non_zero_coords = np.argwhere(image > 0)

    if non_zero_coords.size == 0:
        raise ValueError("The image does not contain any non-zero pixels.")

    # Extract x and y coordinates
    y_coords, x_coords = non_zero_coords[:, 0], non_zero_coords[:, 1]

    # Get bounding box coordinates
    xmin, xmax = x_coords.min(), x_coords.max()
    ymin, ymax = y_coords.min(), y_coords.max()

    # Calculate the center of the object
    x_center = xmin + (xmax - xmin) / 2
    y_center = ymin + (ymax - ymin) / 2

    center = [x_center, y_center]
    return center

# Rotated the aligned point image based on image center, find the optimal angle that have the maximum overlap with the object mask
def test_rotation(object_mask, aligned_points_mask, rotation_center):
    rows, cols = aligned_points_mask.shape
    max_overlap = 0
    best_angle = 0
    optimal_rotate_image = None
    a = np.sum((aligned_points_mask) > 0)

    for angle in range(0, 360, 1):  # Rotate from 0 to 180 degrees in steps of 10
        M = cv2.getRotationMatrix2D(rotation_center, angle, 1)
        rotated_points = cv2.warpAffine(aligned_points_mask, M, (cols, rows), flags=cv2.INTER_NEAREST)
        _, rotated_points = cv2.threshold(rotated_points, 127, 255, cv2.THRESH_BINARY)
        overlap = np.sum((object_mask & rotated_points) > 0)
        #print(a, overlap, angle)

        if max_overlap < overlap:
            max_overlap = overlap
            best_angle = angle
            optimal_rotate_image = rotated_points
            #print(a, overlap, angle)
    return optimal_rotate_image, max_overlap, best_angle

# Align the roated point image in a short range x(-bias_range ,bias_range) and y(-bias_range, bias_range) to further optimal the image alignment
def test_bias(object_mask, points_mask, bias_range=20):
    rows, cols = points_mask.shape
    max_overlap = 0
    optimal_bias = (0, 0)
    optimal_moved_image = None
    a = np.sum((points_mask) > 0)
    #print(a)

    # Test different biases
    for dx in range(-bias_range, bias_range + 1):
        for dy in range(-bias_range, bias_range + 1):
            # Apply the bias
            M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
            translated_points = cv2.warpAffine(points_mask, M_translate, (cols, rows))

            # Calculate the overlap
            overlap = np.sum((object_mask & translated_points) > 0)

            # Update the maximum overlap and optimal bias
            if overlap > max_overlap:
                max_overlap = overlap
                optimal_bias = (dx, dy)
                optimal_moved_image = translated_points

    return optimal_moved_image, max_overlap, optimal_bias

def update_csv_by_filename(file_path, filename, tx, ty, optimal_angle):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            # Create the file and write the header
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Filename', 'tx', 'ty', 'optimal_angle'])
            print(f"File created with header: {file_path}")
        
        # Read the existing data
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            rows = [row for row in reader]
        
        # Ensure the header has the correct columns
        column_count = len(header)
        if header != ['Filename', 'tx', 'ty', 'optimal_angle']:
            raise ValueError("The CSV file has an incorrect header.")

        # Check if the filename exists and update or append the data
        updated = False
        for row in rows:
            if row[0] == filename:
                # Extend the row to match the number of header columns
                while len(row) < column_count:
                    row.append('')
                
                # Update the relevant columns
                row[1] = tx
                row[2] = ty
                row[3] = optimal_angle
                updated = True
        
        # Append the row if no match is found
        if not updated:
            rows.append([filename, tx, ty, optimal_angle])
            print(f"New row added for {filename}.")

        # Write the updated data back to the file
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Data updated successfully in {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def align_loop(object_image_file, converted_image, image_file_name, debug_dir, processed_dir, csv_file_name):
    # Set output path
    file_name = image_file_name
    debug_image_path = os.path.join(debug_dir, file_name)
    processed_points_image_path = os.path.join(processed_dir, file_name)
    csv_file_path = os.path.join(processed_dir, "result.csv")

    # Load the object mask
    object_image = cv2.imread(object_image_file, cv2.IMREAD_GRAYSCALE)
    object_edges = cv2.Canny(object_image, 50, 200)
    object_mask = cv2.threshold(object_image, 50, 255, cv2.THRESH_BINARY)[1]
    object_mask_edge = cv2.threshold(object_edges, 50, 255, cv2.THRESH_BINARY)[1]

    # Load and preprocess the points image
    points_image = converted_image
    points_mask = cv2.threshold(points_image, 235, 255, cv2.THRESH_BINARY)[1]

    # Align centers, rotate, and find optimal bias
    aligned_points_mask, translation, centroid = align_centers(points_mask)

    # Initialize optimal angle, translation and overlap
    optimal_angle = 0
    optimal_translation = translation
    print(optimal_translation)
    current_overlap = 0
    max_overlap = 10

    while(current_overlap < max_overlap):
        current_overlap = max_overlap
        optimal_rotate_image, max_overlap, best_angle = test_rotation(object_mask, aligned_points_mask, centroid)
        
        if 181 <= best_angle <= 359:
            best_angle = best_angle - 360
        

        if (current_overlap < max_overlap):
            optimal_angle += best_angle
            rows, cols = points_image.shape
            M_rotate = cv2.getRotationMatrix2D(centroid, best_angle, 1)  # Rotation
            rotated_points_image = cv2.warpAffine(points_image, M_rotate, (cols, rows))

        aligned_points_mask, max_overlap, optimal_bias = test_bias(object_mask, optimal_rotate_image)
        if (current_overlap < max_overlap):
            optimal_translation[0] += optimal_bias[0]
            optimal_translation[1] += optimal_bias[1]
            print(optimal_translation)
            M_translate = np.float32([[1, 0, optimal_bias[0]], [0, 1, optimal_bias[1]]])  # Bias
            aligned_points_image = cv2.warpAffine(rotated_points_image, M_translate, (cols, rows))

        centroid = calculate_object_center(aligned_points_mask)

    processed_points_image = aligned_points_image
    # Overlay the object mask and the modified points mask for debugging
    debug_image = cv2.cvtColor(object_mask_edge, cv2.COLOR_GRAY2BGR)
    debug_image[:, :, 1] = np.maximum(debug_image[:, :, 1], optimal_rotate_image)

    # Save the debug image
    cv2.imwrite(debug_image_path, debug_image)
    # Save the processed points image
    cv2.imwrite(processed_points_image_path, processed_points_image)

    # Convert the processed image to a DataFrame
    df = pd.DataFrame(processed_points_image)
    # Save the DataFrame to a CSV file
    csv_path = os.path.join(processed_dir, f"{file_name}.csv")
    df.to_csv(csv_path, index=False, header=False)

    print(f"Processed {file_name}")

     # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image File Name', 'Optimal Angle', 'dx', 'dy'])
        # Write the results to the CSV file
        csv_writer.writerow([file_name, optimal_angle, optimal_translation[0], optimal_translation[1]])
    
    update_csv_by_filename(csv_file_name, os.path.splitext(file_name)[0], optimal_translation[0], optimal_translation[1], optimal_angle)

    return processed_points_image_path, processed_points_image

def stl_to_greyscale(input_file, data_path):
    file_name_no_ext, _ = os.path.splitext(os.path.basename(input_file))
    side, identifier = file_name_no_ext.split('_', 1)
    output_folder = "Output"
    converted_dir = os.path.join(data_path, output_folder, "Raw", identifier)
    debug_dir = os.path.join(data_path, output_folder, "Debug", identifier)
    aligned_dir = os.path.join(data_path, output_folder, "Aligned", identifier)
    # Create output directory if it doesn't exist
    os.makedirs(converted_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)
    csv_file_name = os.path.join(data_path, output_folder, "processed_results.csv")

    object_size, converted_image, image_file_name = STL_to_png(input_file, converted_dir)
    str_value = str(int(object_size[1] / 10) + 1)
    object_file_name = side + "_" + str_value + ".png"
    object_image_file = os.path.join(data_path, "Data", "Insole", side, object_file_name)

    aligned_image_path, processed_points_image = align_loop(object_image_file, converted_image, image_file_name, debug_dir, aligned_dir, csv_file_name)
    return aligned_image_path, processed_points_image

# Example usage when called from another script
if __name__ == "__main__":
    input_file = r"D:\A\Test\Test1\Input\left_01wangchongguang.stl"
    data_path = r"D:\A\Test\Test1"
    aligned_image_path, processed_points_image = stl_to_greyscale(input_file, data_path)