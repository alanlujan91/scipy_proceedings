# multinterp comments, figures and notebooks  

Please feel free to incorporate into the paper or ignore at your discretion. :-) -- Jennifer Yoon --

Screenshot PNG files with hand drawn markups are stored in ` .\figures\`  

Curvenotes build preview with animation (figures) look very nice, cool!  

## Table of Contents, Jupyter Notebooks  
(in order of appearance in paper)

### 1. Multivariate_Interpolation.ipynb  

cells 3 and 4:   
"Suppose we are trying to approximate the following function at a set of points: "
```
def squared_coords(x, y):
    return x**2 + y**2
```
comment: It will be helpful to provide a short reason for choosing the squared coordinates function. [[ example: A squared x and y coordinates function is used to draw a figure whose grid geometry looks like a curved sheet in 3D projection. ]]

[[This closed-form solution function, for which all points along the curve is known, is used as the baseline, to sample points that will simulate an unknown function. Then we use one of the interpolation methods to calculate points in between sampled grid locations of the similarly shaped curve function.]]

First 2 figures need titles and descriptions.  
Blue colors are too uniform. Is there a way to have gradation?, so curve will be exaggerated?  

It's not obvious what is different in the 2 figures. They look identical. May benefit from having a small boxed area where the pixelation is greatly enhanced in the interpolated output,figure 2.  

[[Title: figure 1: 3D projection of squared coordinates function. ]]

screenshot3.PNG <img src="\figures\screenshot3.PNG" width="600" >


[[Title figure 2: interpolated 3D projection, using sampled points from figure 1. Image is pixelated because outputs are interpolations.]]

screenshot4.PNG <img src="\figures\screenshot4.PNG" width="600" >



### 2. Multivariate_Interpolation_with_Derivatives.ipynb   

Difficult to see relationship between first group of 2 figures and second group of 2 figures (partial derivatives).  

axis need labels on all 4 figures.  

screenshot5.PNG <img src="\figures\screenshot5.PNG" width="600" >

### 3. Multivalued_Interpolation.ipynb

none. Could use grid lines or pop-out box with enhanced pixelation to make the differences between plots easier to eye-ball.  


### 4. Curvilinear_Interpoliation.ipynb 

none. Could use grid lines or pop-out box with enhanced pixelation to make the differences between plots easier to eye-ball.  


### 5. Unstructured_Interpolation.ipynb


    For figures after the 1st one, (group 1: nearest, linear, cubic, radial basis) and (group 2: original, gaussian process regression), eye-balling the differences in the images maybe easier with grid lines drawn in white or black ink. 
    
    Or a small boxed area can be blown up and pixelation exaggerated in the interpolated image for viewing contrast. 

    Say something like, "Boxed area's pixelation has been enhanced on the simulated (interpolated) figure to maximize visual difference. Image does not reflect smoothness of model's output.  



screenshot1.PNG  <img src="\figures\screenshot1.PNG" width="700" >

screenshot2.PNG <img src="\figures\screenshot2.PNG" width="700" >
