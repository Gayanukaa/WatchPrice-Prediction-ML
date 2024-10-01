# Scripting the Process and Model Deployment

## Files Included
- `config.py`: Configuration settings
- `data.py`: Data preprocessing logic
- `model.py`: Model definition and related logic
- `predict.py`: Prediction logic
- `train.py`: Model training script
- `requirements.txt`: List of required Python libraries

## How to Use

### 1. Install Required Libraries
First, install the required dependencies:
```bash
pip install -r requirements.txt
```
### 2. Train the Model
Run the following command to train the model:

```bash
python train.py
```
### 3. Make Predictions
After training, you can make predictions by running:
```bash
python predict.py
```
### 4. Access FastAPI Swagger UI
Once the application is running, you can access the FastAPI Swagger UI for testing at: ```http://127.0.0.1:8000/docs```

## Docker Usage

To build and start the application using the following command:

```bash
docker run -p 80:80 watchprice-fastapi
```

<!-- 1. Create a Requirements File
Generate the requirements.txt file for the Docker image:

```bash
pip list --format=freeze > requirements.txt
```

2. Build the Docker Image
To build the Docker image, run:

```bash
docker build -t watchprice-fastapi .
```

3. Run the Docker Container
Finally, start the application using the following command:

```bash
docker run -p 80:80 watchprice-fastapi
``` -->

## Notes
- In this project, initially used the `pandas.get_dummies()` function to transform categorical variables. However, this method is not always ideal, especially when there is a large number of categories. <br><br> The issue with `pandas.get_dummies()` is that it converts all categorical variables during training. However, when making predictions, the model might encounter new values in the categorical variables. The model won't recognize these values because they are part of the dimension created during the initial transformation. <br><br> To overcome this, use **Scikit-learn's OneHotEncoder**, which handles unseen categories during prediction more effectively.

- This project leverages FastAPI for building APIs and Docker for containerization. Make sure Docker is installed and running on your system before building and running the container.
