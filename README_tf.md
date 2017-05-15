# Using GPU acceleration with tensorflow for the DTCWT

Normally you would import the dtcwt library and set up a forward transform.
E.g.
```python
import dtcwt
t = dtcwt.Transform2d(biort='near_sym_a',qshift='qshift_b')
p = t.forward(X, nlevels)
low, highs = p.lowpass, p.highpasses
```
To use the tensorflow acceleration, you must however import the specific
module, and use the slightly modified functions. E.g.
```python
import dtcwt.tf 
t = dtcwt.tf.Transform2d(biort='near_sym_a',qshift='qshift_b')
p = t.forward(X, nlevels)
low, highs = p.lowpass, p.highpasses
```
In this instance, X was a numpy array. The library will create all the ops on
the graph, and feed the numpy array into it, create a session and evaluate it.
This provides little advantage over the straightforward numpy operation, but is
there for compatability.

For real speed-up, you want to feed batches of images into the library. An
example would be:
```python
import dtcwt.tf
t = dtcwt.tf.Transform2d(biort='near_sym_a',qshift='qshift_b')
imgs = tf.placeholder(tf.float32, [None, 100,100])
p = t.forward(imgs, nlevels)
low_op, high_ops = p.lowpass_op, p.highpasses_ops
sess = tf.Session()
low = sess.run(low_op, {imgs:X})
```
Having to evaluate each op independently would be quite annoying, so I've made
a helpful routine for it, called eval_fwd
```python
import dtcwt.tf
t = dtcwt.tf.Transform2d(biort='near_sym_a',qshift='qshift_b')
imgs = tf.placeholder(tf.float32, [None, 100,100])
p_tf = t.forward(imgs, nlevels) # returns a dtcwt.Pyramid_tf object
sess = tf.Session()
X = np.random.randn(10,100,100)
p = p_tf.eval_fwd(X) # returns a dtcwt.Pyramid object
lows, highs = p.lowpass, p.highpasses
assert lows.shape[0] == 10
```
In this example, the returned pyramid object, p, now has a batch of lowpass and
highpasses.

For added help, the forward transform can also accept channels of inputs (where
the regular dtcwt only accepts single channel input) through a special module
called forward_channels. At this point, it is likely you will not be wanting to
handle a pyramid, so instead this function returns a tuple of tensors. The tuple will be
formed of:

(lowpass, (highpass[0], highpass[1], ... highpass[nlevels-1])),

or if the include_scale option is true, then:

(lowpass, (highpass[0], highpass[1], ... highpass[nlevels-1])),
 (scale[0], scale[1], ... scale[nlevels-1]))

i.e.
```python
import dtcwt.tf
t = dtcwt.tf.Transform2d(biort='near_sym_a',qshift='qshift_b')
imgs = tf.placeholder(tf.float32, [None, 100,100,3])
yl,yh,yscale  = t.forward_channels(imgs, nlevels,include_scale=True)
sess = tf.Session()
X = np.random.randn(10,100,100,3)
lows = sess.run(yl, {imgs:X})
```



