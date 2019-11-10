# Brain-Computer Interface workspace

My objective here is to share some of the code, models, and data from the OpenBCI 16-channel headset. I suspect many people are not going to be able to get their hands on the headset, but that doesn't mean you can't still play with some of the data! 

# Files
`training.py` - This is merely an example of training a model with this data. I have yet to find any truly great model, though at the end of this readme, I will try to keep an updated confusion matrix of my best-yet models. This way, you can easily tell if you've been able to find something better than what I've got. 

If people are able to beat my model and are willing to share their models. I will post some sort of highscores somewhere on this repo.

Since some people wont be able to resist making a model on validation data...I will use my own separate validation data to actually create scores. If you're not cheating, this shouldn't impact you ;)

`analysis.py` - You can use this to run through validation data to see confusion matricies for your models on out of sample data.

# Requirements
Numpy
TensorFlow (I am using 2.0, but I am not using anything 2.0 specific)

# The data

Currently, the data available is 16-channel FFT 0-60Hz, sampled at a rate of about 25/second.

I am not sure where I want to put the data, but, for now, it's available here: https://hkinsley.com/static/downloads/bci/model_data.7z

*File structure*: 
<ul>
	<li>data
		<ul>
			<li>left</li>
			<li>none</li>
			<li>right</li>
		</ul>
	</li>
	<li>validation_data
		<ul>
			<li>left</li>
			<li>none</li>
			<li>right</li>
		</ul>
	</li>
</ul>
 
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


