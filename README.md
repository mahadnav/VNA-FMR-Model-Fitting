# VNA Model Fitting & Analysis Notebook

## Introduction

This Jupyter Notebook provides a comprehensive analysis of data obtained from VNA (Vector Network Analyzer) experiments. The notebook is designed to process and visualize data collected from both 1-port and 2-port experiments, applying various functions for Spectrum fitting, Kittel fitting, and Linear fitting.

## Prerequisites

Ensure that you have the required Python libraries installed. You can install them using the following command:

```bash
pip install numpy pandas matplotlib lmfit
```

## Usage

### Loading Data and Viewing Raw Results

The initial part of the notebook focuses on loading raw experimental data and visualizing it. The script utilizes functions from the `func` and `data_container` modules to process and prepare the data for subsequent analysis.

### Model Development

A fitting model for the FMR (Ferromagnetic Resonance) spectrum is developed using LMfit. The model is defined in the `_dp_dh_script`, and the fitting function is implemented in `dp_dh_model`. This model will be used for fitting the experimental data in later stages.

### Spectrum Fitting

The notebook then proceeds to fit the model to the original data set. It involves looping over frequencies, extracting relevant parameters, creating a model, setting initial parameters, and fitting the model to the experimental data. The fit results are saved, and the fitted plots are generated and saved to an output directory.

### Kittel Fitting

Another fitting analysis is performed using the Kittel equation, providing insights into the gyromagnetic ratio, saturation magnetization, effective magnetization, and uniaxial anisotropy. The results are visualized and compared with the experimental data.

### Linear Fitting

The notebook concludes with a linear fitting analysis, including linear regression, plotting, and displaying regression results. Gilbert damping constant (alpha parameter) is calculated based on the obtained slope, providing additional insights into the material under test.

### FWHM (Full Width Half Maximum)

The FWHM function is applied to analyze the Full Width Half Maximum of the spectrum at specific frequencies. This provides information about the width of resonance peaks.

### Lorentzian Fitting

The Lorentzian fitting function is utilized to fit Lorentzian profiles to the experimental data. This analysis helps in understanding the shape of resonance peaks and extracting relevant parameters.

## Credits

- **Developer:** Mahad Naveed (BS Physics @ LUMS 2023)
- **Supervisor:** Dr. Sabieh Anwar
- **Mentor:** Dr. Adnan Raza

For information about PhysLab, visit: www.physlab.org
