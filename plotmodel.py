from keras.models import load_model
from keras.utils.visualize_util import plot

model = load_model('model.h5')
plot(model, to_file='model.png')