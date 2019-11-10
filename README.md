# Brain-Computer Interface workspace

My objective here is to share some of the code, models, and data from the OpenBCI 16-channel headset. I suspect many people are not going to be able to get their hands on the headset, but that doesn't mean you can't still play with some of the data! 

# The data

Currently, the data available is 16-channel FFT 0-60Hz, sampled at a rate of about 25/second.

I am not sure where I want to put the data, but, for now, it's available here: https://hkinsley.com/static/downloads/bci/model_data.7z

*File structure*: 
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

Each file is targeted to be 10 seconds long, which, at 25 iter/sec gives us, the 250 (though you should not depend/assume all files will be exactly 250 long). Then you have the number of channels (16), and then 60 values, for up to 60Hz. For example, if you do: 

```py
import numpy as np
import matplotlib.pyplot as plt

d = np.load("data/left/1572814991.npy")

plt.plot(d[0][0])
plt.show()
```

You will see a graph of: The data for Channel 0 for the very first sample.
![FFT graph single channel](https://pythonprogramming.net/static/images/bci/fft-single-channel.png)

If you want to see all 16 channels:
```py
import numpy as np
import matplotlib.pyplot as plt

d = np.load("data/left/1572814991.npy")

for channel in d[175]:
	plt.plot(channel)
plt.show()
```
![FFT graph 16 channels](https://pythonprogramming.net/static/images/bci/fft-16-channels.png)


