from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from markupsafe import Markup


file = open('cropmodel2.pkl', 'rb')
svm = pickle.load(file)
file.close()

with open('fertilizer.pkl', 'rb') as file:
    fertilizer_dic = pickle.load(file)

app = Flask(__name__)

mapper = {1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans', 5: 'pigeonpeas', 6: 'mothbeans', 7: 'mungbean', 8: 'blackgram', 9: 'lentil', 10: 'pomegranate', 11: 'banana', 12: 'mango', 13: 'grapes', 14: 'watermelon', 15: 'muskmelon', 16: 'apple', 17: 'orange', 18: 'papaya', 19: 'coconut', 20: 'cotton', 21: 'jute', 22: 'coffee'}

fertilizer_dic = {
    'NHigh': """The N value of soil is high and might give rise to weeds.
        <br/> Please consider the following suggestions:
        <br/><br/> 1. <i> Manure </i> – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.
        <br/> 2. <i>Coffee grinds </i> – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.
        <br/>3. <i>Plant nitrogen fixing plants</i> – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil
        <br/>4. Plant ‘green manure’ crops like cabbage, corn and brocolli
        <br/>5. <i>Use mulch (wet grass) while growing crops</i> - Mulch can also include sawdust and scrap soft woods""",
    'Nlow': """The N value of your soil is low.
        <br/> Please consider the following suggestions:
        <br/><br/> 1. <i>Add sawdust or fine woodchips to your soil</i> – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.
        <br/>2. <i>Plant heavy nitrogen feeding plants</i> – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.
        <br/>3. <i>Water</i> – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.
        <br/>4. <i>Sugar</i> – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. This is similar concept to adding sawdust/woodchips which are high in carbon content.
        <br/>5. Add composted manure to the soil.
        <br/>6. Plant Nitrogen fixing plants like peas or beans.
        <br/>7. <i>Use NPK fertilizers with high N value.
        <br/>8. <i>Do nothing</i> – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops.""",
    'PHigh': """The P value of your soil is high.
        <br/> Please consider the following suggestions:
        <br/><br/>1. <i>Avoid adding manure</i> – manure contains many key nutrients for your soil but typically including high levels of phosphorous. Limiting the addition of manure will help reduce phosphorus being added.
        <br/>2. <i>Use only phosphorus-free fertilizer</i> – if you can limit the amount of phosphorous added to your soil, you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium. Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.
        <br/>3. <i>Water your soil</i> – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.
        <br/>4. Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).
        <br/>5. Use crop rotations to decrease high phosphorous levels""",
    'Plow': """The P value of your soil is low.
        <br/> Please consider the following suggestions:
        <br/><br/>1. <i>Bone meal</i> – a fast acting source that is made from ground animal bones which is rich in phosphorous.
        <br/>2. <i>Rock phosphate</i> – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.
        <br/>3. <i>Phosphorus Fertilizers</i> – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).
        <br/>4. <i>Organic compost</i> – adding quality organic compost to your soil will help increase phosphorous content.
        <br/>5. <i>Manure</i> – as with compost, manure can be an excellent source of phosphorous for your plants.
        <br/>6. <i>Clay soil</i> – introducing clay particles into your soil can help retain & fix phosphorus deficiencies.
        <br/>7. <i>Ensure proper soil pH</i> – having a pH in the 6.0 to 7.0 range has been scientifically proven to have the optimal phosphorus uptake in plants.
        <br/>8. If soil pH is low, add lime or potassium carbonate to the soil as fertilizers. Pure calcium carbonate is very effective in increasing the pH value of the soil.
        <br/>9. If pH is high, addition of appreciable amount of organic matter will help acidify the soil. Application of acidifying fertilizers, such as ammonium sulfate, can help lower soil pH""",
    'KHigh': """The K value of your soil is high</b>.
        <br/> Please consider the following suggestions:
        <br/><br/>1. <i>Loosen the soil</i> deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.
        <br/>2. <i>Sift through the soil</i>, and remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.
        <br/>3. Stop applying potassium-rich commercial fertilizer. Apply only commercial fertilizer that has a '0' in the final number field. Commercial fertilizers use a three number system for measuring levels of nitrogen, phosphorous and potassium. The last number stands for potassium. Another option is to stop using commercial fertilizers all together and to begin using only organic matter to enrich the soil.
        <br/>4. Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.
        <br/>5. Use NPK fertilizers with low K levels and organic fertilizers since they have low NPK values.
        <br/>6. Grow a cover crop of legumes that will fix nitrogen in the soil. This practice will meet the soil’s needs for nitrogen without increasing phosphorus or potassium.
        """,
    'Klow': """The K value of your soil is low.
        <br/>Please consider the following suggestions:
        <br/><br/>1. Mix in muricate of potash or sulphate of potash
        <br/>2. Try kelp meal or seaweed
        <br/>3. Try Sul-Po-Mag
        <br/>4. Bury banana peels an inch below the soils surface
        <br/>5. Use Potash fertilizers since they contain high values potassium
        """
}

# Add a dictionary to map crops to fertilizers
crop_fertilizer = {
    'rice': 'Urea',
    'maize': 'Di-ammonium Phosphate',
    'chickpea': 'Molybdenum-based fertilizer',
    'kidneybeans': 'Molybdenum-based fertilizer',
    'pigeonpeas': 'Di-ammonium Phosphate',
    'mothbeans': 'Phosphate-based fertilizer',
    'mungbean': 'Di-ammonium Phosphate',
    'blackgram': 'Di-ammonium Phosphate',
    'lentil': 'Rhizobium inoculants',
    'pomegranate': 'Potash',
    'banana': 'NPK',
    'mango': 'Potassium',
    'grapes': 'NPK',
    'watermelon': 'Nitrogen-rich fertilizer',
    'muskmelon': 'NPK',
    'apple': 'NPK',
    'orange': 'NPK',
    'papaya': 'NPK',
    'coconut': 'NPK',
    'cotton': 'Urea',
    'jute': 'NPK',
    'coffee': 'NPK'
}

# Add a dictionary to map crops to water requirements
crop_water = {
    'rice': '500-700 mm',
    'maize': '400-600 mm',
    'chickpea': '300-400 mm',
    'kidneybeans': '300-400 mm',
    'pigeonpeas': '300-400 mm',
    'mothbeans': '200-300 mm',
    'mungbean': '200-300 mm',
    'blackgram': '200-300 mm',
    'lentil': '200-300 mm',
    'pomegranate': '600-800 mm',
    'banana': '1200-2200 mm',
    'mango': '600-800 mm',
    'grapes': '500-700 mm',
    'watermelon': '400-600 mm',
    'muskmelon': '400-600 mm',
    'apple': '600-800 mm',
    'orange': '600-800 mm',
    'papaya': '1000-1500 mm',
    'coconut': '1500-2500 mm',
    'cotton': '700-1300 mm',
    'jute': '500-700 mm',
    'coffee': '1500-2500 mm'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/water_predict')
def water_predict():
    return render_template('water_predict.html')

@app.route('/water_result', methods=['POST'])
def water_result():
    district = request.form['district']
    crop = request.form.get('crop', 'Unknown Crop')  # Assuming crop is provided in the form
    irrigation_method = request.form.get('irrigation_method', 'Unknown Method')  # Assuming irrigation method is provided in the form
    
    # Define water consumption parameters for different crops and irrigation methods
    water_consumption_params = {
        'drip': {
            'rice': 1000,
            'maize': 600,
            'chickpea': 300,
            'kidneybeans': 300,
            'pigeonpeas': 300,
            'mothbeans': 200,
            'mungbean': 200,
            'blackgram': 200,
            'lentil': 200,
            'pomegranate': 500,
            'banana': 1500,
            'mango': 500,
            'grapes': 400,
            'watermelon': 400,
            'sugarcane': 400,
            'apple': 500,
            'orange': 500,
            'papaya': 1000,
            'coconut': 1500,
            'cotton': 800,
            'jute': 500,
            'coffee': 1500,
            'chilli': 400,
            'onion': 400,
            'oilseeds': 300,
            'cumin': 200,
            'coriander': 200,
            'wheat': 400,
            'mustard': 300,
            'fennel': 200
        },
        'sprinkler': {
            'rice': 1200,
            'maize': 800,
            'chickpea': 400,
            'kidneybeans': 400,
            'pigeonpeas': 400,
            'mothbeans': 300,
            'mungbean': 300,
            'blackgram': 300,
            'lentil': 300,
            'pomegranate': 700,
            'banana': 1800,
            'mango': 700,
            'grapes': 600,
            'watermelon': 500,
            'sugarcane': 500,
            'apple': 700,
            'orange': 700,
            'papaya': 1200,
            'coconut': 2000,
            'cotton': 1000,
            'jute': 600,
            'coffee': 2000,
            'chilli': 600,
            'onion': 500,
            'oilseeds': 400,
            'cumin': 300,
            'coriander': 300,
            'wheat': 500,
            'mustard': 400,
            'fennel': 300
        },
        'flood': {
            'rice': 1500,
            'maize': 1000,
            'chickpea': 500,
            'kidneybeans': 500,
            'pigeonpeas': 500,
            'mothbeans': 400,
            'mungbean': 400,
            'blackgram': 400,
            'lentil': 400,
            'pomegranate': 900,
            'banana': 2200,
            'mango': 900,
            'grapes': 800,
            'watermelon': 600,
            'sugarcane': 600,
            'apple': 900,
            'orange': 900,
            'papaya': 1500,
            'coconut': 2500,
            'cotton': 1300,
            'jute': 700,
            'coffee': 2500,
            'chilli': 800,
            'onion': 600,
            'oilseeds': 500,
            'cumin': 400,
            'coriander': 400,
            'wheat': 600,
            'mustard': 500,
            'fennel': 400
        },
        'canal': {
            'rice': 1400,
            'maize': 900,
            'chickpea': 450,
            'kidneybeans': 450,
            'pigeonpeas': 450,
            'mothbeans': 350,
            'mungbean': 350,
            'blackgram': 350,
            'lentil': 350,
            'pomegranate': 800,
            'banana': 2000,
            'mango': 800,
            'grapes': 700,
            'watermelon': 550,
            'sugarcane': 550,
            'apple': 800,
            'orange': 800,
            'papaya': 1300,
            'coconut': 2200,
            'cotton': 1100,
            'jute': 650,
            'coffee': 2200,
            'chilli': 700,
            'onion': 550,
            'oilseeds': 450,
            'cumin': 350,
            'coriander': 350,
            'wheat': 550,
            'mustard': 450,
            'fennel': 350
        }
    }
    # Get the water consumption for the selected crop and irrigation method
    crop_params = water_consumption_params.get(irrigation_method.lower(), {})
    water_consumption = crop_params.get(crop.lower(), 'Unknown')  # Default to 'Unknown' if method not found

    if water_consumption == 'Unknown':
        water_consumption = 'Unknown liters/day m3/hectre'
    else:
        water_consumption = f"{water_consumption} cubic litre/hectre"

    return render_template('water_result.html', district=district, crop=crop, irrigation_method=irrigation_method, water_consumption=water_consumption)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        mydict = request.form
        try:
            input_features = [float(mydict.get('nitrogen')), float(mydict.get('phosphorus')), float(mydict.get('potassium')),
                              float(mydict.get('temperature')), float(mydict.get('humidity')), float(mydict.get('ph')), float(mydict.get('rainfall'))]

            inf = svm.predict([input_features])[0]
            value = mapper.get(inf, None)

            if value is None:
                return render_template('error.html', message="Model prediction is invalid.")

            df = pd.read_csv('fertilizer.csv')
            crop_data = df[df['Crop'] == value].iloc[0]

            n_diff = int(crop_data['N']) - int(input_features[0])
            p_diff = int(crop_data['P']) - int(input_features[1])
            k_diff = int(crop_data['K']) - int(input_features[2])

            diffs = {'N': n_diff, 'P': p_diff, 'K': k_diff}
            max_diff = max(diffs, key=lambda x: abs(diffs[x]))
            key = f"{max_diff}{'High' if diffs[max_diff] < 0 else 'low'}"

            response = Markup(str(fertilizer_dic[key]))
            value = value.capitalize()

            # Get the recommended fertilizer for the crop
            recommended_fertilizer = crop_fertilizer.get(value.lower(), 'No specific fertilizer recommendation')

            # Get the water requirement for the crop
            water_requirement = crop_water.get(value.lower(), 'No specific water requirement')

            return render_template('result.html', inf=response, value=value, fertilizer=recommended_fertilizer, water=water_requirement)
        except Exception as e:
            return render_template('error.html', message=str(e))

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
    app.run(debug=True)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')