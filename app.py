#This is Heroku Deployment Lectre
from flask import Flask, request, render_template
import os
import pickle

print("Test")
print(os.getcwd())
path = os.getcwd()

with open('Models/DT_Model.pkl', 'rb') as f:
    DT = pickle.load(f)

with open('Models/KNN_Model.pkl', 'rb') as f:
    KNN = pickle.load(f)

with open('Models/GNB_Model.pkl', 'rb') as f:
    GNB_model = pickle.load(f)


def get_predictions(age, sex, chest_pain_type, resting_blood_pressure, serum_cholestoral, fasting_blood_sugar,resting_electrocardiographic_results, maximum_heart_rate_achieved, exercise_induced_angina, oldpeak,the_slope_of_the_peak_exercise_ST_segment, number_of_major_vessels, thal, req_model):
    mylist = [age, sex, chest_pain_type, resting_blood_pressure, serum_cholestoral, fasting_blood_sugar,resting_electrocardiographic_results, maximum_heart_rate_achieved, exercise_induced_angina, oldpeak,the_slope_of_the_peak_exercise_ST_segment, number_of_major_vessels, thal]
    mylist = [float(i) for i in mylist]
    vals = [mylist]

    if req_model == 'DecisionTree':
        #print(req_model)
        return DT.predict(vals)[0]

    elif req_model == 'KNearestNeighbour':
        #print(req_model)
        return KNN.predict(vals)[0]

    elif req_model == 'NaiveBayes':
        #print(req_model)
        return GNB_model.predict(vals)[0]
    else:
        return "Cannot Predict"


app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        chest_pain_type = request.form['chest_pain_type']
        resting_blood_pressure = request.form['resting_blood_pressure']
        serum_cholestoral = request.form['serum_cholestoral']
        fasting_blood_sugar = request.form['fasting_blood_sugar']
        resting_electrocardiographic_results = request.form['resting_electrocardiographic_results']
        maximum_heart_rate_achieved = request.form['maximum_heart_rate_achieved']
        exercise_induced_angina = request.form['exercise_induced_angina']
        oldpeak = request.form['oldpeak']
        the_slope_of_the_peak_exercise_ST_segment = request.form['the_slope_of_the_peak_exercise_ST_segment']
        number_of_major_vessels = request.form['number_of_major_vessels']
        thal = request.form['thal']
        req_model = request.form['req_model']

        target = get_predictions(age, sex, chest_pain_type, resting_blood_pressure, serum_cholestoral, fasting_blood_sugar,resting_electrocardiographic_results, maximum_heart_rate_achieved, exercise_induced_angina,oldpeak,the_slope_of_the_peak_exercise_ST_segment, number_of_major_vessels, thal, req_model)

        if target==1:
            diagnosis = 'Patient is like to have a heart disease'
        else:
            diagnosis = 'Patient is unlike to have a heart disease'

        return render_template('home.html', target = target, sale_making = diagnosis)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)