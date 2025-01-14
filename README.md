# SmartInsoleGUI
### Script Description

This script processes 3D foot models in STL format and converts them into greyscale images for further analysis. It includes steps to handle image alignment, optimization, and conversion to CSV format for numerical analysis. Below is a detailed overview:

---

#### **Key Functionalities**
1. **STL to PNG Conversion**:
   - Converts STL files into greyscale PNG images representing height as pixel intensity.
   - Allows resolution scaling for high-detail images.
   - Post-processes images to fill gaps and improve the quality of the greyscale representation.

2. **Alignment and Optimization**:
   - Aligns the object's center to the image center.
   - Rotates and translates the image to maximize overlap with a reference object.
   - Fine-tunes alignment through small translations.

3. **Output Management**:
   - Saves intermediate and final processed images to specific directories.
   - Logs alignment and optimization details in a CSV file.
   - Exports aligned images and data for further processing.

4. **Helper Functions**:
   - Samples points within 3D triangles for accurate image generation.
   - Applies filters to smooth images and fill gaps.
   - Updates CSV logs dynamically with results.

---

#### **Inputs**
- `input_file`: The path to the STL file to be processed.
- `data_path`: The root directory where input data is located and output will be saved.

---

#### **Outputs**
- Aligned greyscale PNG image in the specified `aligned` directory.
- Debug images showing overlays of object masks and alignment steps in the `debug` directory.
- CSV files containing alignment parameters and processed image data.

---

#### **How to Use**
1. **Prepare Input Data**:
   - Place your STL file in the appropriate input directory.

2. **Run the Script**:
   - Call the script with the desired `input_file` and `data_path`:
     ```bash
     python STL_process.py
     ```

3. **Check Outputs**:
   - Greyscale PNG images will be saved under `Output/Raw` and `Output/Aligned`.
   - Debug and alignment details will be saved under `Output/Debug`.
   - Processed data logs will be in `Output/processed_results.csv`.

4. **Customization**:
   - Modify resolution scaling in `STL_to_png` if higher/lower resolution is required.
   - Adjust alignment sensitivity and overlap threshold in alignment functions.

---

#### **Example Execution**
```python
if __name__ == "__main__":
    input_file = r"D:\A\Test\Test1\Input\left_01wangchongguang.stl"
    data_path = r"D:\A\Test\Test1"
    aligned_image_path = stl_to_greyscale(input_file, data_path)
    print(f"Aligned image saved at: {aligned_image_path}")
```

This will process the specified STL file, align it with the reference object, and save all outputs in structured folders.
