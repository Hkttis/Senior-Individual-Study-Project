
# Reconstructing the Map of the Han Dynasty's Western Regions Using Force-Directed Algorithms

*Individual Science Research Project (Grade 12) by Lin Yun-Han*

## Project Overview

This project aims to reconstruct the geographic layout of approximately 35 states in the Western Regions during the Han Dynasty, using a **force-directed algorithm** and spatial data derived from historical Chinese texts (*Book of Han*, *Shiji*, etc.). The reconstructed map serves as a reference for identifying possible locations of ancient ruins and proposes a reproducible algorithm applicable to other historical reconstructions.

The work extends a previous Grade 11 project based on Multidimensional Scaling (MDS), by integrating **physical simulation with attractive (spring) and repulsive (electrostatic-like) forces**. It also includes **error visualization** and **quantitative evaluation** of reconstruction accuracy.

## Key Features

- Applies a **force-directed layout algorithm** using Pymunk to model physical interactions.
- Uses **directional constraints and historical distances** between ancient states from *Book of Han*.
- Introduces **fixed anchors** and **pre-simulation alignment** to ensure geographic stability.
- Provides **quantitative accuracy metrics**: RMSE and Kruskal’s stress.
- Produces **heatmaps and overlay plots** to compare reconstructed maps with ground truth.

## Main Technologies

- Python, Pymunk, Pygame
- Geographic projection with Lambert Conformal Conic (LCC)
- CSV-based structured data
- Physics-based graph optimization

## How to Run

### Dependencies

- Python 3.8+
- `numpy`, `pymunk`, `pygame`, `pyproj`, `csv`, `math`

### Execution

Run the main simulation with:

```bash
python spring.py
```

The script will:
- Load historical distance and direction data
- Generate an initial layout
- Perform a physics-based layout simulation
- Output stress convergence plots, error heatmaps, and map overlays

## Evaluation Results

| Model Version      | RMSE (km) | Kruskal's Stress |
|--------------------|-----------|------------------|
| Grade 11 (MDS)     | 1308.84   | 0.8788           |
| Grade 12 (Spring)  | **450.34**| **0.0103**        |

The Grade 12 model significantly improves accuracy, both visually and quantitatively.

## Visualization Outputs

- Stress convergence plot (log-scaled)
- Directional error heatmap with zoom-in
- Overlay comparison between reconstructed and historical map (LCC projected)

## Acknowledgements

- Prof. Tsung-Han Tsai, Academia Sinica  
- Mr. Wei-Li Pan, Jianguo High School  
- Based on the work of Chen Shih-Liang’s mileage hypothesis  
- Historical source data: *Book of Han*, *Shiji*, *Hou Hanshu*
