We used pandas dummy module to transform for categorical variables. But it's doesn't always work especially when you have higher number of categories. <br><br>
Pandas dummy sees all the variables and converts. But during prediction we pass only one at a time. Model will not know what the other values are. Because other values are also in dimensions.<br><br>
Instead we use sklearn one-hot encodings.

Files included
- config.py : Configuration file
- data.py: Data preprocessing file
- model.py: Model file
- predict.py: Prediction file
- train.py: Training file
- requirements.txt: Required libraries

## How to use
1. Install required libraries
```bash
pip install -r requirements.txt
```
2. Run the training file
```bash
python train.py
```
3. Run the prediction file
```bash
python predict.py
```