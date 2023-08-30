# HISTOGRAM
# Histogram and Histogram Equalization of an image
## Aim
To obtain a histogram for finding the frequency of pixels in an Image with pixel values ranging from 0 to 255. Also write the code using OpenCV to perform histogram equalization.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Load the image using a suitable library like OpenCV.

### Step2:
Convert the image to grayscale using cvtColor() if needed.

### Step3:
Calculate the histogram using calcHist() and plot it with Matplotlib.

### Step4:
Perform histogram equalization using equalizeHist() for enhanced contrast.

### Step5:
Visualize results by plotting histograms and displaying original and equalized images side by side.

## Program:

# Developed By:PRANEET S
# Register Number:212221230078
```
For a grayscale image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a grayscale image
image_path = 'G.jpg'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Calculate histogram
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Plot the histogram
plt.plot(hist)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of Grayscale Image')
plt.show()
```
```
For colour image

import cv2
import numpy as np
import matplotlib.pyplot as plt

#### Load a color image
image_path = 'M.jpg'
color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

#### Convert color image to grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

#### Calculate histogram
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

#### Plot the histogram
plt.plot(hist)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of color Image')
plt.show()
```





# Display the histogram of gray scale image and any one channel histogram from color image
```
Histogram of Grayscale image

import cv2
import numpy as np
import matplotlib.pyplot as plt

#### Load a grayscale image
image_path = 'mi.jpg'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#### Calculate histogram
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

#### Plot the histogram
plt.plot(hist)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of Grayscale Image')
plt.show()

```

# Write the code to perform histogram equalization of the image. 
```

import cv2
import numpy as np
import matplotlib.pyplot as plt

#### Load a grayscale image
image_path = 'im.jpg'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#### Perform histogram equalization
equalized_image = cv2.equalizeHist(gray_image)

#### Calculate histograms before and after equalization
hist_before = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_after = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

#### Plot the histograms
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(hist_before)
plt.title('Original Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(2, 1, 2)
plt.plot(hist_after)
plt.title('Equalized Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

#### Display the original and equalized images
cv2.imshow('Original Grayscale Image', gray_image)
cv2.imshow('Equalized Grayscale Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## Output:
### Input Grayscale Image and Color Image
![image](https://github.com/Prakashmathi2004/HISTOGRAM/assets/118350045/fcb74ba3-2347-4a14-beeb-29ba0415933c)

![image](https://github.com/Prakashmathi2004/HISTOGRAM/assets/118350045/1991a34c-bc58-46c4-b9c0-564b6eaab512)

### Histogram of Color Image
<img width="438" alt="image" src="https://github.com/Prakashmathi2004/HISTOGRAM/assets/118350045/131d5b0b-1747-4bcd-aedf-fe46436701ab">

### Histogram of Pixel intensities
![image](https://github.com/Prakashmathi2004/HISTOGRAM/assets/118350045/2df92f31-ba53-4285-ad99-d5eceba029af)


### Histogram of Grayscale Image and any channel of Color Image
![image](https://github.com/Prakashmathi2004/HISTOGRAM/assets/118350045/f736d812-90be-43be-902f-9fa6b1f3c7a6)

![image](https://github.com/Prakashmathi2004/HISTOGRAM/assets/118350045/d05da7cc-4c9c-46a8-ad28-c8cbe94b866a)


### Histogram Equalization of Grayscale Image
<img width="574" alt="image" src="https://github.com/Prakashmathi2004/HISTOGRAM/assets/118350045/e504dadc-e3c8-425d-bfea-81475e61f835">

## Result: 
Thus the histogram for finding the frequency of pixels in an image with pixel values ranging from 0 to 255 is obtained. Also,histogram equalization is done for the gray scale image using OpenCV.
