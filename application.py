from flask import Flask, render_template,request
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)

# Define the path to the CSV file
csv_file_path = 'Luxury watch clean.csv'

@app.route('/')
def index():
    try:
        # Load the CSV file into a DataFrame
        watch = pd.read_csv('Luxury watch clean.csv')
        brand =sorted(watch['Brand'].unique())
        models = sorted(watch['Model'].unique())
        casemat = sorted(watch['Case Material'].unique())
        strap = sorted(watch['Strap Material'].unique())
        movement = sorted(watch['Movement Type'].unique())
        waterrezis = sorted(watch['Water Resistance'].unique())
        casediam = sorted(watch['Case Diameter (mm)'].unique())
        casethick = sorted(watch['Case Thickness (mm)'].unique())
        bandwid = sorted(watch['Band Width (mm)'].unique())
        dialcol = watch['Dial Color'].unique()
        crystalmat = watch['Crystal Material'].unique()
        complicattion = watch['Complications'].unique()
        powerres = sorted(watch['Power Reserve'].unique())

        

        # Render the template with the extracted attributes
        return render_template('index.html',brand=brand, models=models, casemat=casemat,
                               strap=strap, movement=movement, waterrezis=waterrezis,
                               casediam=casediam, casethick=casethick, bandwid=bandwid,
                               dialcol=dialcol, crystalmat=crystalmat, complicattion=complicattion,
                               powerres=powerres)
    except Exception as e:
        # If an error occurs, render an error template
        return render_template('error.html', error=str(e))

model=pickle.load(open('./LinearRegressionModell.pkl','rb'))

@app.route('/predict',methods=['POST'])
def predict():
    brand = request.form.get('brand')
    modell = request.form.get('watch_models')
    case_material = request.form.get('case_material')
    strap_material = request.form.get('strap_material')
    movement_type = request.form.get('movement_type')
    water_resistance = np.log(float(request.form.get('water_resistance')))
    case_diameter = np.log(float(request.form.get('case_diameter')))
    case_thickness = np.log(float(request.form.get('case_thickness')))
    band_width = np.log(float(request.form.get('band_width')))
    dial_color = request.form.get('dial_color')
    crystal_material = request.form.get('crystal_material')
    complications = request.form.get('complications')
    power_reserve = np.log(float(request.form.get('power_reserve')))

    prediction=model.predict(pd.DataFrame([[brand,modell,case_material,strap_material,movement_type,water_resistance,case_diameter,case_thickness,band_width,dial_color,crystal_material,complications,power_reserve]],columns=['Brand', 'Model', 'Case Material', 'Strap Material', 'Movement Type', 'Water Resistance', 'Case Diameter (mm)', 'Case Thickness (mm)', 'Band Width (mm)', 'Dial Color', 'Crystal Material', 'Complications', 'Power Reserve']))
    print(prediction)
    return str(int(round(prediction[0],0)))
if __name__ == '__main__':
    app.run(debug=True)
