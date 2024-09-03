import os
import numpy as np
import matplotlib.pyplot as plt

def load_all_gt_files(base_gt_data_dir):
    """
    Load all ground truth .npy files from the subdirectories of the base directory.

    Parameters:
    - base_gt_data_dir: The base directory containing subdirectories with .npy files.

    Returns:
    - List of all ground truth .npy file paths, sorted by filename.
    """
    gt_files = []
    for root, _, files in os.walk(base_gt_data_dir):
        for file in sorted(files):
            if file.endswith('.npy'):
                gt_files.append(os.path.join(root, file))
    return gt_files

def plot_comparison_curve(gt_npy_file, output_npy_file, save_folder, point, file_name):
    """
    Plot and compare the intensity curves from ground truth and network output.

    Parameters:
    - gt_npy_file: Path to the ground truth .npy file.
    - output_npy_file: Path to the network output .npy file.
    - save_folder: Folder where the comparison curve will be saved.
    - point: Tuple (x, y) representing the point in the image.
    - file_name: The base name of the input file for saving the plot.
    """
    # Load the ground truth and network output .npy files
    gt_image = np.load(gt_npy_file).squeeze()
    output_image = np.load(output_npy_file).squeeze().transpose((1, 2, 0))

    # Ensure the image shapes are compatible
    assert gt_image.shape[2] == 31, "Expected 31 channels in the ground truth image."
    assert output_image.shape[2] == 31, "Expected 31 channels in the output image."

    # Extract the pixel values at the given point
    x, y = point
    gt_intensity_values = gt_image[y, x, :]
    output_intensity_values = output_image[y, x, :]

    # Normalize the intensity values
    gt_intensity_normalized = gt_intensity_values / np.max(gt_intensity_values)
    output_intensity_normalized = output_intensity_values / np.max(output_intensity_values)

    # Define the wavelength range
    wavelengths = np.linspace(400, 700, 31)

    # Plot the curves
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, gt_intensity_normalized, 'b-', label='Ground Truth (Normalized)')
    plt.plot(wavelengths, output_intensity_normalized, 'r-', label='Network Output (Normalized)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity')
    plt.title(f'Comparison at Point ({x}, {y})')
    plt.legend()
    plt.grid(True)

    # Create the save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save the plot using the original input filename as the base name
    save_path = os.path.join(save_folder, f'{file_name}_comparison_curve.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Comparison curve saved at: {save_path}")

def main():
    base_gt_data_dir = '/home/wangruozhang/Hyper2Mosaic/Dataset/Extracted_Data'
    save_npy_folder = '/home/wangruozhang/mosaic2hsi/TestOutput1/npy'
    comparison_save_folder = '/home/wangruozhang/mosaic2hsi/ComparisonOutput'

    # Load all ground truth files
    gt_files = load_all_gt_files(base_gt_data_dir)

    # Sort output files to match the order of gt_files
    output_files = sorted(os.listdir(save_npy_folder))

    # Ensure that the number of ground truth files matches the number of output files
    if len(gt_files) != len(output_files):
        print(f"Mismatch in number of files: {len(gt_files)} GT files and {len(output_files)} output files.")
        return

    # Process each pair of ground truth and output .npy files
    for gt_file, output_file in zip(gt_files, output_files):
        gt_filename = os.path.basename(gt_file)
        base_filename = os.path.splitext(gt_filename)[0]

        output_npy_file = os.path.join(save_npy_folder, output_file)

        # Check if the output file exists and proceed to plot the comparison
        if os.path.exists(output_npy_file):
            plot_comparison_curve(gt_file, output_npy_file, comparison_save_folder, point=(630, 630), file_name=base_filename)
        else:
            print(f"Missing output file for: {gt_file}")

if __name__ == "__main__":
    main()
