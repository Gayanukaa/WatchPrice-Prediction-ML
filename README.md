# Fitness-Trackers-EDA

End to End Linear Regression Project

Dataset - [Fitness Trackers Products Ecommerce](https://www.kaggle.com/datasets/arnabchaki/fitness-trackers-products-ecommerce)

## Overview

<p>
  <em>Developed with the software and tools below.</em>
</p>
<p>
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=python&logoColor=yellow" alt="Python">
<img src="https://img.shields.io/badge/Conda-44A833.svg?style=flat&logo=anaconda&logoColor=white" alt="Anaconda">
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=jupyter&logoColor=white" alt="Jupyter">
<img src="https://img.shields.io/badge/Scikit_Learn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="Scikit Learn">
<img src="https://img.shields.io/badge/FastAPI-009688.svg?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=docker&logoColor=white" alt="Docker">


</p>


## Project Lifecycle

Separate sections tested out in jupyter notebooks. Final scripts and project deployment available in [src](src).

1. Problem Identification
2. Business Understanding
3. Collecting Data
4. Pre-Processing Data
5. [Analyzing Data](main.ipynb)
6. [Data Handling](feature_engineering.ipynb)
7. [Model Evaluation/Monitoring](model_building.ipynb)
8. Model Training
9. Model Deployment

## Project Structure

```
|—— data
|    |—— clean.csv
|    |—— smartwatches.csv
|—— feature_engineering.ipynb
|—— main.ipynb
|—— model.pkl
|—— model_building.ipynb
|—— src
|    |—— Dockerfile
|    |—— Overview.md
|    |—— data
|        |—— clean.csv
|        |—— smartwatches.csv
|        |—— test.json
|    |—— dummys
|        |—— Brand
|        |—— Dial Shape
|        |—— Model Name
|        |—— Strap Material
|        |—— numerical_col
|    |—— models
|        |—— model.pkl
|    |—— main.py
|    |—— config.py
|    |—— data.py
|    |—— predict.py
|    |—— requirements.txt
|    |—— test.py
|    |—— train.py
|    |—— test.json
```
## Results

| Model Stage | R2 Score | Standard Deviation | Hyperparameters                                           |
|------------------------------|-----------------------------------------------|--------------------|---------------------------------------------------------------|
| Decision Tree Model                 | 0.1336        | 0.2665             | -             | -                                                           |
| Random Forest Model                 | 0.5850        | 0.1444             | -             | -                                                           |
| XGB Model               | 0.6697    | 0.1146             | -             | -                                                             |
| After Hyperparameter Tuning (XGB model)   | 0.7615                                             | -             |  learning_rate: 0.2 <br> max_depth: 3<br> n_estimators: 300   |


### Tested Platform
```
OS: Ubuntu 22.04.4 LTS
Python: 3.9.6
```
<!-- ## Installation

1. Clone the repository: `git clone https://github.com/Gayanukaa/Fitness-Trackers-EDA.git`
2. Navigate to the project directory: `cd Fitness-Trackers-EDA/src`
3. Set up the environment with dependencies: `conda env create -f environment.yml -n dspy-dev`
4. Activate the environment required for the application: `conda activate dspy-dev`
5. Run the application: `streamlit run app.py` -->

## License

This project is licensed under the [MIT License](LICENSE).

<!-- ## Commands

conda activate ml-pyt
conda deactivate -->