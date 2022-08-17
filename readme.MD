# Gender Prediction Model based on First Name
This prediction model is created based on character level LSTM model built using `tensorflow` with `python3.8`.

- [Simple EDA done on raw data](https://huggingface.co/spaces/jamesnzeex/gender_prediction/blob/main/src/EDA.ipynb)
<p align = "center">
    <img src="./chart.png" width = 800>
</p>

## Deployment
To set up the app, clone the repo and install the requirements. 
```
git clone https://huggingface.co/spaces/jamesnzeex/gender_prediction

# change to project repo
cd gender_prediction/src

# install requirements
pip install -r requirements.txt

# to predict gender given first name of person
python serve.py -n "Alan" "Bobby" "Celine"
Output:
{"Alan":"Male","Bobby":"Male","Celine":"Female"}

# via flask API
python serve.py -d
# send curl command
curl -X POST -d '{"name": ["Alan", "Bobby", "Celine"]}' http://127.0.0.1:5000/ -H 'Content-Type:application/json'
Output:
{"Alan":"Male","Bobby":"Male","Celine":"Female"}

# to retrain model
python serve.py -r
```