import tensorflow as tf
import tensorflow_transform as tft 
from tensorflow.keras import layers
import os  
import tensorflow_hub as hub
from tfx.components.trainer.fn_args_utils import FnArgs
 
LABEL_KEY = "Weather_Type"
 
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"
 
def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
 
 
def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs,
             batch_size=64)->tf.data.Dataset:
    """Get post_transform feature & create batches of data"""
    
    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    
    def format_data(features, labels):
        labels = tf.reshape(labels, [-1, 4])
        return features, labels

    return dataset.map(format_data)


def model_builder():
    """Build model"""
    # Input layer
    inputs = []
    inputs.append(tf.keras.Input(shape=(1,), name=transformed_name('Temperature')))
    inputs.append(tf.keras.Input(shape=(1,), name=transformed_name('Humidity')))
    inputs.append(tf.keras.Input(shape=(1,), name=transformed_name('Wind_Speed')))
    inputs.append(tf.keras.Input(shape=(1,), name=transformed_name('Precipitation')))
    inputs.append(tf.keras.Input(shape=(1,), name=transformed_name('Atmospheric_Pressure')))
    inputs.append(tf.keras.Input(shape=(1,), name=transformed_name('UV_Index')))
    inputs.append(tf.keras.Input(shape=(1,), name=transformed_name('Visibility')))
    inputs.append(tf.keras.Input(shape=(1,), name=transformed_name('Cloud_Cover')))
    inputs.append(tf.keras.Input(shape=(1,), name=transformed_name('Season')))
    inputs.append(tf.keras.Input(shape=(1,), name=transformed_name('Location')))
    
    # Concatenate input layers
    x = tf.keras.layers.Concatenate()(inputs)
    
    # Hidden layers
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        
        feature_spec = tf_transform_output.raw_feature_spec()
        
        feature_spec.pop(LABEL_KEY)
        
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        transformed_features = model.tft_layer(parsed_features)
        
        # get predictions using the transformed features
        return model(transformed_features)
        
    return serve_tf_examples_fn

def _get_transform_features_signature(model, tf_transform_output):
  """Returns a serving signature that applies tf.Transform to features."""

  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
  model.tft_layer_eval = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def transform_features_fn(serialized_tf_example):
    """Returns the transformed_features to be fed as input to evaluator."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_eval(raw_features)
    return transformed_features

  return transform_features_fn

def run_fn(fn_args: FnArgs):
    """Train the model"""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    # Get train & eval data
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=1)
    
    # Build model
    model = model_builder()
    
    # Train model
    model.fit(train_dataset, epochs=10, validation_data=eval_dataset)
    
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                                    tf.TensorSpec(
                                    shape=[None],
                                    dtype=tf.string,
                                    name='examples')),
        'transform_features':
        _get_transform_features_signature(model, tf_transform_output),
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
