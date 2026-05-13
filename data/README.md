# Data Organization Format

This document explains the required data organization format and requirements for the HybridBathNet project.

## Data Folder Structure

Data should be organized according to the following hierarchical structure:

```text
HybridBathNet/data/
├── Island_A/
│   ├── 1.tif
│   ├── 2.tif
│   ├── 3.tif
│   └── ...
├── Island_B/
│   ├── 1.tif
│   ├── 2.tif
│   ├── 3.tif
│   └── ...
└── Island_C/
    ├── 1.tif
    ├── 2.tif
    ├── 3.tif
    └── ...
```

## Data Organization Requirements

### 1. Folder Organization
- **Data for each island** should be placed in the same folder
- All `.tif` files for the same island must be **of the same size (blocks)**
- Folder names should clearly identify the corresponding island

### 2. TIF File Format Requirements

Each `.tif` file should contain multiple bands with the following requirements:

#### Band Organization Structure
```text
Band 1: Satellite imagery data (Band 2 - Blue)
Band 2: Satellite imagery data (Band 3 - Green) 
Band 3: Satellite imagery data (Band 4 - Red)
...
Band n-1: WDI data (Water Depth Index, second-to-last band)
Band n:   Ground truth depth data (last band)
```

#### Data Requirements
- **Ground Truth Depth Data**:
  - Located in the **last band** of the `.tif` file
  - Non-water areas (such as land regions) should be set to `NaN`
  - Depth values should be actual depth measurements

- **WDI Data**:
  - Located in the **second-to-last band** of the `.tif` file
  - Should be **pre-computed**, not calculated during training

- **Satellite Imagery Data**:
  - Typically uses RGB bands (Red, Green, Blue)
  - Corresponds to specific bands from Landsat/Sentinel satellites

## Data Preprocessing Notes

1. **Data Consistency**: Ensure that all files within the same island have consistent spatial resolution and coordinate reference system
2. **NaN Handling**: The code will automatically handle NaN values by replacing them with 0 and generating corresponding masks
3. **Data Type**: Recommend using `float32` data type to save memory while maintaining precision
4. **File Naming**: Suggest using sequential numeric naming (e.g., 1.tif, 2.tif, ...) for easier data loading

## Optional Depth Normalization

If normalization is required for the water depth data, the nonlinear normalization method proposed by Sun et al. (2023) can be used.

Reference:

```text
Sun, S., Chen, Y., Mu, L., Le, Y., & Zhao, H. (2023). Improving shallow water bathymetry inversion through nonlinear transformation and deep convolutional neural networks. Remote Sensing, 15(17), 4247. https://doi.org/10.3390/rs15174247
```

For positive water depth values \(h \ge 0\), the normalization function is:

$$
y =
\begin{cases}
\dfrac{h^2}{20}, & 0 \le h \le 2 \\[6pt]
\dfrac{h}{h+8}, & h > 2
\end{cases}
$$

where \(h\) is the original water depth value and \(y\) is the normalized depth value.
