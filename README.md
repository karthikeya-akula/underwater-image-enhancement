# Underwater Image Enhancement

## Project Overview
This project implements traditional underwater image enhancement techniques using Python and OpenCV.

## Techniques Implemented
- Histogram Equalization
- CLAHE
- Gamma Correction
- White Balance Correction

## Metrics Used
- PSNR
- SSIM
- UIQM

## Features
- Enhancement pipeline
- Comparison table
- Visual comparison
- Logging
- Modular code structure

## Folder Structure
- data/ → input images
- outputs/ → enhancement results
- metrics/ → evaluation results
- src/ → source code
- config/ → configuration settings

## Project Structure

```text
underwater-enhancement/
│
├── config/
│   └── config.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── logs/
│   └── project.log
│
├── metrics/
│   └── comparison_results.csv
│
├── notebooks/
│   └── enhancement_demo.ipynb
│
├── outputs/
│   ├── clahe/
│   ├── comparison_visuals/
│   ├── gamma/
│   ├── histogram_eq/
│   └── white_balance/
│
├── src/
│   ├── enhancement.py
│   ├── logger.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   └── utils.py
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Project

```bash
python main.py
```

## Author

Karthikeya Akula