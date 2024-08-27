# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import math

"""# Problem 1 (20 Points)

Link to input image: https://drive.google.com/file/d/1X0xir8CrOoKsNXxvZnLVAJWKztpMuUhV/view?usp=drive_link

#### Part 1: Write code and describe the steps to perform histogram equalization of the given color image.  Hint: First convert to a color space that separates the intensity from the color channels (eg. LAB or LUV color spaces work the best).  Your result should look like the below:

Link to the output of part-1: https://drive.google.com/file/d/1nIhoE0PyCdFE3LfAwGcUqUt_GaQ2NoG-/view?usp=sharing

"""

# Image is read from the path
image1 = cv2.imread('PA120272.JPG')

# The BGR image is conbverted to LAB format
# Here L represents the Lightness/Intensity whereas A and B respresnts the colors.
# We need to perform histogram equalization on the L channel
lab = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
L,A,B=cv2.split(lab)

# Histogram Linearization is performed on the L channel
histogram_equalized_L = cv2.equalizeHist(L)

# The equalized L channel is merged back with color channels
equalized_image=cv2.merge((histogram_equalized_L,A,B))

# The LAB image is converted to RGB image for plotting
plot_image1=cv2.cvtColor(equalized_image, cv2.COLOR_LAB2RGB)
cv2.imwrite('midterm_q1_eq.jpg', plot_image1)

"""#### Part 2: Another way to enhance a dark image is to perform gamma correction.   For each intensity pixel, replace it with the new intensity = 255*((intensity/255)^(1/2.2)). Write code to perform gamma correction.  Your result should look like below:

Link to the output of part-2: https://drive.google.com/file/d/1rgo7Kl8qK7Byh5QZsIb4CrfdBzY51Aco/view?usp=sharing

"""

# The BGR image is conbverted to LAB format
# Here L represents the Lightness/Intensity whereas A and B respresnts the colors.
# Gamma correction is performed on the L channel
gamma_corrected_L = 255*((L/255)**(1/2.2))
gamma_corrected_L_int=np.uint8(gamma_corrected_L)

# The gamma corrected channel is merged back with color channels
gamma_image = cv2.merge((gamma_corrected_L_int,A,B))

# The LAB image is converted to RGB image for plotting
plot_image2=cv2.cvtColor(gamma_image, cv2.COLOR_LAB2RGB)
cv2.imwrite('midterm_q1_gamma.jpg', plot_image2)

"""# Problem 2 (15 Points)

#### 1. For an image that is n by n and assuming that we have a random kernel call it H, where H is m by m matrix, what is the big O notation when performing convolution (how many multiplications are needed, in terms of n)?

#### 2. Describe the meaning of “separable” kernel? Redo the previous question knowing that the H kernel is separable?

#### 3. Apply the principle of separability to the following kernel to separate it into two kernels.


Kernel

<img src="https://drive.google.com/uc?id=1MMCS5erDiy2Pk_ytuyI_GSI8FdjGRUib" width="300" height="200" />

#### Problem-2 (1)

While performing convolution, the convolution kernel is slided over the image such that each pixel of the image is at the centre of the kernel. At each pixel, the kernal values are multiplied with corresponding pixel values. Hence, if we have a m * m kernel, at each pixel, m * m multiplication is done. For example, if the size of the kernel is 3 * 3, 9 multiplications are done at each pixel. If the size of the image is n * n pixels, the multiplication (convolution) is done n * n times (for each pixel). Hence the total number of multiplications will be m * m * n * n or ( (m ^ 2) * (n ^ 2) ). Therefore the Big O notation is O( (m ^ 2)*(n ^ 2) )

Note: To perform convolution on edge pixels, padding the image is necessary. In my explanation, it is assumed that the image is padded.

#### Problem-2 (2)

A seperable kernel is a kernel which can be split/seperated into 2 smaller kernerls. The multiplication of one of the smaller kernel with the other will give us the original kernel. One of the smaller kernel (row kernel) is used for performing row multiplication and the other (column kernel) one is used for performing column multiplication, for convolution. If the size of the original kernel is m * m , the 2 smaller kernels will have the size of m * 1 and 1 * m. For example, if we have a 3 * 3 kernel, if it is seperable, it can be split into two smaller 3 * 1 and 1 * 3 kernels.

While performing convolution, instead of performing m * m multiplications at each pixel, m row multiplications are performed on each pixel and m column multiplications are performed on each pixel of the n * n image. Hence the number of multiplications will be ((m * n * n)+(m * n * n)) = (2 * m * (n^2)). Hence the Big O notation of this will be O(m * (n^2))

By comparing the BigO notation, we can observe that seperable kernels are more efficient with respect to computation time.

Note: To perform convolution on edge pixels, padding the image is necessary. In my explanation, it is assumed that the image is padded.

#### Problem-2 (3)

Applying the principle of separability, the given kernel can be split into 2 smaller kernels. The 2 smaller kernels are:

row kernel = (1/4) * Transpose([1 , 2 , 1 ])

column kernel = (1/4) * [1 , 2 , 1 ]

If the row kernel is multiplied with the column kernel, we get the original kernel.
"""

original_kernel = np.array([[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]])

# Row Kernel is a column vector
row_kernel = np.array([[1/4],[2/4],[1/4]])

# Column Kernel is a row vector
column_kernel = np.array([1/4,2/4,1/4])
kernel = row_kernel * column_kernel
are_equal = np.array_equal(original_kernel, kernel)

# Conditional ststement to check if the computed kernel by multiplying row and column kernel is equal to thegiven kernel
if are_equal:
  print("The kernel is: ",'\n', kernel )

"""# Problem 3 (20 Points)

Link to the csv file: https://drive.google.com/file/d/1LGG32bpU0sTIp-lOxOXCZJELGoZqVaZk/view?usp=sharing

#### Given x, y, z coordinates of n number of data points(problem3_points.csv). Consider matrix A where matrix A represents the coordinates of the n points shown in the figure below. Let’s say we want to fit these data points onto an ellipse.

#### 1. Describe the steps to find the major and minor axis of this ellipse along with their prospective length.

#### 2. Similarly, given the data that represents a flat ground in front of the robot in 3D, try to extract the equation of the surface that describe the ground.

#### Problem-3 (1)

In the given problem, there are datapoints in 3d space and these datapoints have to be fit into a 2D ellipse. This can be achieved by following these steps:

1. Computing covariance matrix:

  A covariance matrix is calculated for the 3d points. This matrix will be in the form of
  
  Cov_matrix = [ [ cov(x,x) , cov(x,y) , cov(x,z) ] , [ cov(y,x) , cov(y,y) , cov(y,z) ] , [ cov(z,x) , cov(z,y) , cov(z,z) ] ]

  where cov(x,y) = ( 1 / n ) * ( Sum ( ( xi - mean(x) ) * Transpose( yi - mean(y) ) ) )

2. Extracting Eigen Values and Eigen Vectors of the covariance matrix:

  This is achieved by performing eighen value decomposition n the covariance matrix.

3. Computing the major and minor axis of the ellipse:

  The eigen vector of the covariance matrix associated with the largest eigen value represents the direction of maximum varience. Hence it represents the major axis of the ellipse fit. Similarly, the eigen vector corresponding to the smallest eigen value represents the direction of minimum variance. Hence, it represents the minor axis of the ellipse fit. The largest and the smallest eigen values can be used to compute the prospective lengths of the major and the minor axes of the ellipse fit.

  The direction of the major and minor axis of the ellipse gives the orientation of the ellipse in 3d space.


Alternative approach:

1. Fit the 3d data points into a plane.
2. Find the rotation matrix to rotate this plane so that it matches with a coordinate plane (for example xy plane)
3. Use this roatation matrix to rotate the datapoints so that they now (approximately) align with a coordinate plane
4. Now, the 3d datapoints are converted to 2d datapoints (an axis is eliminated. For example, if the points are projected/aligned on the xy plane, the z axis component of the points will be eliminated), which makes the curve fitting easy.
5. The datapoints can then be fitted into a ellipse by using least square method.
6. The points along with the fit can be rotated in the resverse direction (with respect to the rotation matrix computed above) to match the original distribution of the data.
7. The major and minor axis of the ellipse can be computed from the parameters of the fit.

#### Problem-3 (2)

The equation of the plane is given by:

( a * x ) + ( b * y ) + c = z

By considering this equation in matrix form, we get:

[ [ xi ] , [ yi ] , [ 1 ] ] * [ [ a ] , [ b ] , [ c ] ] = [ zi ]

(where xi, yi, zi represent the column vector of x , y and z datapoints and 1 represent a column vecor of ones)

Let matrix Axy = [ [ xi ] , [ yi ] , [ 1 ] ]

and matrix Bz = [ zi ]

Now taking the left pseudo inverse of matrix Axy and multiplying it with Bz gives the values of the coefficients a , b and c of the plane (by LHS = RHS concept)
"""

# The csv file is read using numpy
csv_data = np.loadtxt("problem3_points.csv", delimiter=",",skiprows=1)

# The x,y and z coordinate values are extracted from the csv file
x = csv_data[:, 0]
y = csv_data[:, 1]
z = csv_data[:, 2]

# Creating an empty Axy matrix and filling the values as per the explanation above
Axy = np.empty((x.shape[0], 3))
Axy[:,0]=x[:]
Axy[:,1]=y[:]
Axy[:,2]=1

# Creating an empty Bz matrix and filling the values as per the explanation above
Bz=np.empty((z.shape[0], 1))
Bz[:,0]=z[:]

# The left Pseudo inverse of a matrix M is given by ((Transpose(M)*M)^-1)*(Transpose(M)
# Left Pseudoinverse of matrix Axy is computed
mat1 = np.matmul(np.transpose(Axy),Axy)
mat2 = np.linalg.inv(mat1)

# Left Pseudoinverse of matrix Axy is multiplied with matrix Bz to get the coefficients
mat3 = np.matmul(np.transpose(Axy),Bz)
coeff = np.matmul(mat2,mat3)

# Printing the coefficient values
print("The plane coefficients are: ",'\n', "a = ",coeff[0],'\n', "b = ",coeff[1],'\n', "c = ",coeff[2])

# Plotting the original dataset and the fitted plane
plt1=plt.figure()
ax1 = plt.subplot(111, projection='3d')
ax1.scatter(x, y, z, color='r',s=3)
# Creating a meshgrid to plot the fitted plane
x_fit,y_fit = np.meshgrid(np.arange(-5, 6),np.arange(-5, 5))
z_fit = np.empty(x_fit.shape)
for row in range(x_fit.shape[0]):
  for column in range(x_fit.shape[1]):
    # From the equation of the plane and the obtained coefficients
    z_fit[row,column] = coeff[0] * x_fit[row,column] + coeff[1] * y_fit[row,column] + coeff[2]

ax1.plot_wireframe(x_fit,y_fit,z_fit, color='b',linewidth=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
plt.savefig('midterm_q3_plane.jpg', bbox_inches='tight')

"""# Problem 4 (30 Points)

Link to the input image: https://drive.google.com/file/d/1VeZyrPIwyg7sqi_I6N5zBUJP9UHkyDEb/view?usp=sharing

#### Given the photo of a train track, describe the process to find the mean distance in pixels between the train tracks for every row of the image where train tracks appear. This question contains two parts:

#### 1. Design Pipeline and explain the requirement of each step
#### 2. Write code to implement the pipeline

#### Problem-4 (1)

Pipeline for computing mean distance between the train tracks:

1. Import the image and declare the source points and the destination points with respect to the image features:
  
  Source points are the current pixel positions and the destination points are the desired pixel positions with respect to an image feature. In this case, we need to get the top view of the train tracks to compute the average distance between the tracks. But, in the image, the tracks apprear to converge due to the phenomenon of vanishing point. Hence, by obtaining the top view of the tracks, we can compute the average distance between the parallel tracks.

2. Perform Perspective transform with respect to the source and destination points to obtain the Transformation matrix.
3. Warp the image using the computed transformation matrix to obtain the top view image of the track.
4. Compute straight lines from the warped image using the concept of HoughLines:

  The warped image is first converted to greyscale, blurred and segmented to seperate out the track part. The edges are then detected using canny edge detector and the houghlines are computed.

5. The distance between the Houghlines are measured to get the average distance between the tracks:

  The distance is calculated and the short distances are ignored. These short distances represent the distance between the two edges of the the same track. Hence, it should be ignored to get the distance between two tracks.

#### Problem-4 (2)
"""

# The image of the train tracks is read and is converted to RGB format for plotting using matplotlib
train_image = cv2.imread('train_track.jpg')
plt_img = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)

# Based on the image plotted and by taking the help of the grid lines, the source and destination points are considered.
# Source points are the points on the track in the given image
# Destination points are the points that the tracks need to be at in the top view
src_pts=np.float32([[1410,1150],[930,2000],[2100,2000],[1610,1150]])
dst_pts=np.float32([[930,1150],[930,2000],[2100,2000],[2100,1150]])

# # The considered source and destination images are plotted for visualization.
for i in src_pts:
    cv2.circle(plt_img, (int(i[0]), int(i[1])), 15, (0, 0, 255), -1)
for i in dst_pts:
    cv2.circle(plt_img, (int(i[0]), int(i[1])), 15, (255, 0, 0), -1)

cv2.imwrite('midterm_q4_points.jpg', cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR))

# A transformation matrix representing a transformation between source and destination points is obtained by using getPerspectiveTransform function.
transform=cv2.getPerspectiveTransform(src_pts,dst_pts)

# Based on the transformation matrix obtained, the image is warpedto get the top view of the tracks
warp= cv2.warpPerspective(train_image,transform,(train_image.shape[1],train_image.shape[0]))

# The warped image is converted to grayscale
gray_train = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

# The grayscale image is blurred to only retain major features and eliminate fine details.
blurred_train = cv2.GaussianBlur(gray_train, (19, 19), 0)

# The blurred image is segmented to extract the pixels which have Grayscale value of more than 240 (white). This represents the tracks.
segmented_train = np.where(blurred_train > 240, 255, 0).astype(np.uint8)

# The edges of the segmented tracks are detected using canny edge detector.
edge_train = cv2.Canny(segmented_train, 50, 100)

# The binary image of the edges is then used to extract the straight lines.
lines = cv2.HoughLines(edge_train, 1, np.pi/180, 269)

# The extracted lines are plotted on the warped image for visualization
if lines is not None:
 for i in range(0, len(lines)):
  rho = lines[i][0][0]
  theta = lines[i][0][1]
  a = math.cos(theta)
  b = math.sin(theta)
  x0 = a * rho
  y0 = b * rho
  pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
  pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
  cv2.line(warp, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)

# cv2_imshow(warp)
warp_plt = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)
cv2.imwrite('midterm_q4_lines.jpg', warp)

# The distance between the lines is computed. As the lines are perpendicular to x axis, the perpendicular distance is computed by btaking the difference between the x coordinate values of the line
perpendicular_dist=[]
for i in range(len(lines)):
  for j in range(i+1, len(lines)):
    theta1 = lines[i][0][1]
    theta2 = lines[j][0][1]
    rho1 = lines[i][0][0]
    rho2 = lines[j][0][0]
    a1 = math.cos(theta1)
    a2 = math.cos(theta2)
    x1 = a1 * rho1
    x2 = a2 * rho2
    p_dist= np.abs(x2 - x1)
    # A condition statement to eliminate short distances. These short distances represent the ditance between the edges of the same track.
    if p_dist>100:
      perpendicular_dist.append(p_dist)

# The outer distance ( distance between the outer edges of the tracks ) and the inner distance ( distance between the inner edges of the tracks ) are considered and the mean is computed.
average_dist=np.mean(perpendicular_dist)

# The mean distance between the tracks is computed and printed out
print("The average distance between the tracks is: ", average_dist, " pixels")

"""# Problem 5 (15 Points)

#### Let’s say you want to design a system that will create a top view of the surrounding around the car, see the image below for referene. Do the following,

(1) Describe and point the location of the sensors on the car in the given picture. Also, show the field of view of the sensors.

(2) Write the pipeline that will take input from these sensors and output a top-view image similar to the one shown below.

Note: Explain the answer in detail and provide necessary justifications for the steps you use in the pipeline.

<img src="https://drive.google.com/uc?id=1USIA11QzqbXxzpkkqOX2qedu3bvhGlDW" width="150" height="250" />

Answer:

To design a system that will create a top view of the surrounding around the car, we need atleast 4 cameras with wide field of view. The 4 cameras are mounted in the front, the rear and the two sides of the car. The field of view of these cameras should be close to 180 degrees or more, to capture the entire surrounding. The below image shows the proposed positions of the camera sensors and their field of view. The red lines respresnt the field of view whereas the blue dots represent the camera positions. The side cameras can also be mounted on the car side mirrors.

From the figure we can observe that when the cameras of wide field of view are placed in the 4 directions of the car, it covers the surroundings of the car. We can also observe that the field of views of the cameras overlap in some regions. This overlap can be used for stitching the images togeather to get the top view or the birds eye view.
"""

"""My Solution : car_camera.jpg')"""

"""Process Pipeline:

1. Extract the images from the camera sensors:

  The images/frames from the camera sensors are extracted.

2. Transform the perspective of the extracted images and warp the perspective to get the top view:

  The perspective of the image is transformed with respect to the certain features, to obtain a transformation matrix. This is achieved by taking a reference and extracting the src points (input points) and defining the dst points (output points). The resulting transformation matrix is used to warp the perspective of the image to get the top view.

3. Perform feature detection on the top-view images using feature detectors like SIFT:

  After getting the top view perspective of each image, features/keypoints of these images are extracted by performing feature detection. This is required to match the features and stitch the images togeather.

4. Match the features/keypoints between adjacent top-view images (front-right, front-left, rear-right, rear-left) using feature matchers like FLANN or Brute force:

  The features/keypoints extracted are then matched between adjacent images and good matches are considered. We should consider only good matches for

5. Perfom Homography based on the matched features/keypoints:

  Based on the matched good features/keypoints, homography matrix is computed.  

6. Stitch the images based on the homography computed to obtain the complete to view of the surrounding of the vehicle:

  The adjacent top view images are then stitched togeather by using the homography computed. The stitched image will give the 360 degree top view image of the car's surrounding.

7. The above steps can be repeated to each frame at a set fps to get a video of the top view.
"""