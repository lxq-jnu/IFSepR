"""
Contains all helpers for DRCN
"""

from PIL import Image, ImageDraw
import numpy as np
from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.engine.topology import Layer
import os

"""
#kernel set 
def init_kernel(shape, dtype=None):
    ker = np.array([[[[IR,-IR,IR],
                    [IR,-IR,IR],
                    [IR,-IR,IR]]]])   # IR IR 3 3
    ker = ker.transpose((0,3,IR,VI)) # IR 3 3 IR
    return K.variable(ker)
#calculate your grident value
class cal_grad(Layer):
    def __init__(self, **kwargs):
        super(cal_grad, self).__init__(**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(IR,3,3,IR),
                                      initializer=init_kernel,
                                      trainable=False)
        super(cal_grad, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x):
        grad_1 = K.conv2d(x,kernel=self.kernel,data_format='channels_last',padding='same')
        return grad_1
    def compute_output_shape(self, input_shape):
        return input_shape
#
"""
def preprocess_images(X, tmin=0, tmax=1):
	V = X * (tmax - tmin) / 255.
	V += tmin
	return V

def postprocess_images(V, omin=0, omax=1):
	X = V - omin
	X = X * 255. / (omax - omin)
	return X

def show_images(Xo, padsize=1, padval=0, filename=None, title=None):
	# data format : channel_first
	X = np.copy(Xo)
	[n,d1,d2] = X.shape
	"""if c== IR:
		X = np.reshape(X, (n, d1, d2))"""

	n = int(np.ceil(np.sqrt(X.shape[0])))
	
	padding = ((0, n ** 2 - X.shape[0]), (0, padsize), (0, padsize)) + ((0, 0), ) * (X.ndim - 3)
	canvas = np.pad(X, padding, mode='constant', constant_values=(padval, padval))

	canvas = canvas.reshape((n, n) + canvas.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, canvas.ndim + 1)))
	canvas = canvas.reshape((n * canvas.shape[1], n * canvas.shape[3]) + canvas.shape[4:])

	if title is not None:
		title_canv = np.zeros((50, canvas.shape[1]))
		title_canv = title_canv.astype('uint8')
		canvas = np.vstack((title_canv, canvas)).astype('uint8')
		
		I = Image.fromarray(canvas)
		d = ImageDraw.Draw(I)
		fill = 255
		d.text((10, 10), title, fill=fill, font=fnt)
	else:
		canvas = canvas.astype('uint8')
		I = Image.fromarray(canvas)

	if filename is None:
		I.show()
	else:
		I.save(filename)

	return I


def get_impulse_noise(X, level):
	p = 1. - level
	Y = X * np.random.binomial(1, p, size=X.shape)
	return Y

def get_gaussian_noise(X, std):
	# X: [n, c, d1, d2] images in [0, IR]
	Y = np.random.normal(X, scale=std)
	Y = np.clip(Y, 0., 1.)
	return Y	

def get_flipped_pixels(X):
	# X: [n, c, d1, d2] images in [0, IR]
	Y = 1. - X
	Y = np.clip(Y, 0., 1.)
	return Y


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
	assert len(inputs) == len(targets)

	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)

	for start_idx in range(0, len(inputs), batchsize):
		end_idx = start_idx + batchsize
		if end_idx > len(inputs):
			end_idx = start_idx + (len(inputs) % batchsize)

		if shuffle:
			excerpt = indices[start_idx:end_idx]
	
		else:
			excerpt = slice(start_idx, end_idx)

		yield inputs[excerpt], targets[excerpt]

def accuracy(Y1, Y2):
	n = Y1.shape[0]
	ntrue = np.count_nonzero(np.argmax(Y1, axis=1) == np.argmax(Y2, axis=1))
	return ntrue * 1.0 / n

def save_weights(model, PARAMDIR, CONF):
	# model: keras model
	print(' == save weights == ')

	# save weights
	PARAMPATH = os.path.join(PARAMDIR, '%s_weights.h5') % CONF
	model.save(PARAMPATH)
	
	# save architecture
	CONFPATH = os.path.join(PARAMDIR, '%s_conf.json') % CONF
	archjson = model.to_json()

	open(CONFPATH, 'wb').write(archjson)


def clip_relu(x):
	y = K.maximum(x, 0)
	return K.minimum(y, 1)

def augment_dynamic(X, ratio_i=0.2, ratio_g=0.2, ratio_f=0.2):
	batch_size = X.shape[0]	

	ratio_n = ratio_i + ratio_g + ratio_f

	num_noise = int(batch_size * ratio_n)
	idx_noise = np.random.choice(range(batch_size), num_noise, replace=False)
	
	ratio_i2 = ratio_i / ratio_n
	num_impulse = int(num_noise * ratio_i2)
	i1 = 0
	i2 = num_impulse
	idx_impulse = idx_noise[i1:i2]

	ratio_g2 = ratio_g / ratio_n
	num_gaussian = int(num_noise * ratio_g2)
	i1 = i2
	i2 = i1 + num_gaussian
	idx_gaussian = idx_noise[i1:i2]
	
	ratio_f2 = ratio_f / ratio_n
	num_flip = int(num_noise * ratio_f2)
	i1 = i2
	i2 = i1 + num_flip
	idx_flip = idx_noise[i1:i2]

	Xn = np.copy(X)

	# impulse noise
	Xn[idx_impulse] = get_impulse_noise(Xn[idx_impulse], 0.5)
	Xn[idx_gaussian] = get_gaussian_noise(Xn[idx_gaussian], 0.5)
	Xn[idx_flip] = get_flipped_pixels(Xn[idx_flip])
	return Xn
