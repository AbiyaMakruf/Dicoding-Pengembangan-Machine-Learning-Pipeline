
#Import library
import tensorflow as tf
import tensorflow_transform as tft

#Mendefinisikan nama label key
LABEL_KEY = "Weather_Type"

#Mendefinisikan nama kolom yang akan di transform
def transformed_name(key):
    return key + '_xf'

#Mendefinisikan fungsi preprocessing
def preprocessing_fn(inputs):
    #Mendefinisikan output dictionary
    outputs = {}

    #Normalisasi fitur numeric
    outputs[transformed_name('Temperature')] = tft.scale_to_z_score(inputs['Temperature'])
    outputs[transformed_name('Humidity')] = tft.scale_to_z_score(inputs['Humidity'])
    outputs[transformed_name('Wind_Speed')] = tft.scale_to_z_score(inputs['Wind_Speed'])
    outputs[transformed_name('Precipitation')] = tft.scale_to_z_score(inputs['Precipitation'])
    outputs[transformed_name('Atmospheric_Pressure')] = tft.scale_to_z_score(inputs['Atmospheric_Pressure'])
    outputs[transformed_name('UV_Index')] = tft.scale_to_z_score(inputs['UV_Index'])
    outputs[transformed_name('Visibility')] = tft.scale_to_z_score(inputs['Visibility'])

    #Encode fitur kategorikal
    outputs[transformed_name('Cloud_Cover')] = tft.compute_and_apply_vocabulary(inputs['Cloud_Cover'])
    outputs[transformed_name('Season')] = tft.compute_and_apply_vocabulary(inputs['Season'])
    outputs[transformed_name('Location')] = tft.compute_and_apply_vocabulary(inputs['Location'])

    #Encode label
    weather_type_indices = tft.compute_and_apply_vocabulary(inputs[LABEL_KEY])
    weather_type_one_hot = tf.one_hot(weather_type_indices, depth=4)
    outputs[transformed_name(LABEL_KEY)] = tf.reshape(weather_type_one_hot, [-1, 4])

    return outputs
