import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Define Camera Intrinsics (Based on iPhone 15 Plus Specs)
# Given values
sensor_width_mm = 8.0  # Estimated sensor width in mm
sensor_height_mm = 6.0  # Estimated sensor height in mm
image_width_px = 6000  # Image width in pixels
image_height_px = 4000  # Image height in pixels
focal_length_mm = 26  # Directly using the given focal length from tech specs

# Convert focal length to pixels
f_x = (focal_length_mm / sensor_width_mm) * image_width_px
f_y = (focal_length_mm / sensor_height_mm) * image_height_px

# Assume principal point is at the image center
c_x = image_width_px / 2
c_y = image_height_px / 2

# Assume skew is 0 for square pixels
s = 0

# Construct intrinsic matrix K
K = np.array([[f_x, s, c_x],
              [0, f_y, c_y],
              [0, 0, 1]])

# Display Intrinsic Matrix
print("Intrinsic Camera Matrix (K):")
print(K)

# Step 2: Load and Process the Checkered Image
# Load the checkerboard image
image_path = "HW2\checkered_square.jpeg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define checkerboard pattern size (number of inner corners)
checkerboard_size = (10, 7)  # 10x7 inner corners
square_size = 25  # mm per square

# Manually define world coordinates of checkerboard points
obj_points = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
obj_points[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

# Find the checkerboard corners
found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
if found:
    # Refine the detected corners for accuracy
    corners_refined = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    
    # Draw and display the corners
    image_with_corners = cv2.drawChessboardCorners(image, checkerboard_size, corners_refined, found)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
    plt.title("Detected Checkerboard Corners")
    plt.show()
    
    # Step 3: Manually Solve for Extrinsic Parameters
    num_points = obj_points.shape[0]
    
    # Convert image points to homogeneous coordinates
    img_points_h = np.hstack((corners_refined.reshape(num_points, 2), np.ones((num_points, 1))))

    # Normalize image coordinates using intrinsic matrix
    img_points_norm = np.linalg.inv(K) @ img_points_h.T
    
    # Solve for homography matrix H (Least Squares)
    A = []
    for i in range(num_points):
        X, Y, _ = obj_points[i]
        u, v, w = img_points_norm[:, i]
        A.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u])
        A.append([0, 0, 0, X, Y, 1, -v * X, -v * Y, -v])
    A = np.array(A)
    
    # Solve for homography using SVD
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    
    # Decompose H into R and t
    H /= H[-1, -1]  # Normalize H
    r1 = H[:, 0]
    r2 = H[:, 1]
    t = H[:, 2]
    
    # Ensure orthonormality of R
    r3 = np.cross(r1, r2)
    R = np.column_stack((r1, r2, r3))
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt  # Ensure it's a valid rotation matrix
    
    # Display Results
    print("Rotation Matrix (R):")
    print(R)
    print("Translation Vector (t):")
    print(t)
    
    # Add the results to the assignment document
    print("\nExtrinsic Parameters:")
    print("Rotation Matrix (R):")
    print(R)
    print("Translation Vector (t):")
    print(t)
    
else:
    print("Checkerboard not detected. Ensure the pattern is fully visible and well-lit.")
