# Brain-Computer Interface workspace

My objective here is to share some of the code, models, and data from the OpenBCI 16-channel headset. I suspect many people are not going to be able to get their hands on the headset, but that doesn't mean you can't still play with some of the data! 

# The data

Currently, the data available is 16-channel FFT 0-60Hz, sampled at a rate of about 25/second.

I am not sure where I want to put the data, but, for now, it's available here: https://hkinsley.com/static/downloads/bci/model_data.7z

File structure: 
-data
  -left
  -none
  -right
-validation_data
  -left
  -none
  -right
 
Contained within the left, none, and right directories are `.npy` files with unix timestamps as their name. Each of the files is a numpy array of shape: 
```py
import numpy as np

d = np.load("data/left/1572814991.npy")
print(d.shape)

>>>(250, 16, 60)
```
