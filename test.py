import cv2
import numpy as np

# Load the extrinsic matrices for the object poses
pose1 = np.loadtxt('Cal_extrinsic0.txt')  # Extrinsic matrix for the first position/orientation
numbers = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14]

for i in numbers:
    pose2 = np.loadtxt(f'Cal_extrinsic{i}.txt')  # Extrinsic matrix for the second position/orientation

    # Extract translation vectors and rotation matrices from the poses
    tvec1 = pose1[:, 3]
    tvec2 = pose2[:, 3]
    rmat1 = pose1[:, :3]
    rmat2 = pose2[:, :3]

    # Calculate the translation difference (distance)
    translation_diff = np.linalg.norm(tvec2 - tvec1)

    # Calculate the rotation difference (angle)
    rotation_diff = cv2.Rodrigues(rmat1.T @ rmat2)[0][0]  # Angle in radians

    # Print the relative position and orientation changes
    print("Relative Position Change (Distance):", translation_diff)
    print("Relative Orientation Change (Angle):", rotation_diff)


    # Load the intrinsic matrix from calibration
    intrinsic_matrix = np.loadtxt('Cal_intrinsic.txt')

    # Define the image coordinates of two points in the image
    x1, y1, z1 = tvec1
    x2, y2, z2 = tvec2
    image_point1 = np.array([[x1, y1]], dtype=np.float32)  # Replace with the actual coordinates
    image_point2 = np.array([[x2, y2]], dtype=np.float32)  # Replace with the actual coordinates

    # Undistort the image points using the intrinsic matrix
    undistorted_point1 = cv2.undistortPoints(image_point1, intrinsic_matrix, None)
    undistorted_point2 = cv2.undistortPoints(image_point2, intrinsic_matrix, None)

    # Calculate the 3D coordinates of the points
    object_point1 = cv2.triangulatePoints(np.eye(3, 4), pose1, undistorted_point1, undistorted_point2)
    object_point2 = cv2.triangulatePoints(np.eye(3, 4), pose2, undistorted_point1, undistorted_point2)

    # Convert homogeneous coordinates to Cartesian coordinates
    object_point1_cartesian = cv2.convertPointsFromHomogeneous(object_point1.T.reshape(-1, 1, 4))[0][0]
    object_point2_cartesian = cv2.convertPointsFromHomogeneous(object_point2.T.reshape(-1, 1, 4))[0][0]

    # Calculate the Euclidean distance between the 3D points
    distance_diff = np.linalg.norm(object_point2_cartesian - object_point1_cartesian)

    # Print the difference in distance
    print("Difference in Distance:", distance_diff)