# Aligning the Unseen in Attributed Graphs: Interplay between Graph Geometry and Node Attributes Manifold
## Code by Anonymous author @ Anonymous institution (2025-2026)

This repository hosts all experiments that support the paper *Aligning the Unseen in Attributed Graphs: Interplay between Graph Geometry and Node Attributes Manifold* from Anonymous authors. Data and models weights are not included due to their size but are available upon reasonable request to the corresponding author.

All the framework is written in Obect-Oriented style and is based on PyTorch.

## How to use this repository
The analysis is divided into three types of files: *.ipynb which are notebooks where you can find all the graphics and results, *.py files in the framework folder where you will find the functions that do the actual lifting (in object oriented style), and *.py in the scripts folder where you can find the running scripts to launch the pipeline on compute units.

**Notebooks:**
+ generateSMS.ipynb: Pipeline that generates the affinity score for the generator of the synthetic dataset
+ generateIDF.ipynb: Pipeline that handles all the files from Ile-de-France Mobilités and INSEE and construct the IDF dataset
+ synthetic.ipynb: The notebook that runs all the expirements and training on the Synthetic dataset
+ idf.ipynb: The notebook that runs all the expirements and training on the IDF dataset
+ baselineSynthetic.ipynb: Pipeline that launches comparable algorithms on our synthetic dataset
+ geodesicComputationExample.ipynb: Some tests to check that all geodesic estimators are behaving as expected

## Licence
CC-BY-NC-SA 4.0 International