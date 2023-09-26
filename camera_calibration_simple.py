# pip install opencv-contrib-python
import cv2
import numpy as np
import os

np.set_printoptions(precision=2, suppress=True)

# input: 사용된 calibration pattern 정보, 촬영된 calibration pattern 사진
wp = 9        # calibration pattern 가로점 수
hp = 6        # calibration pattern 세로점 수
length = 3.3   # calibration pattern 한 변 길이 (mm)
directory = 'C:\\Users\\jayja\\Documents\\Vision\\Project3\\cacal2'
imageExtension = '.jpg'  # 파일 확장자
startImageNum = 0  # 읽기 시작하는 사진 번호
endImageNum = 15   # 읽기를 끝내는 사진 번호 + 1

objp = np.zeros((wp*hp, 3), np.float32)              # 가상의 3차원 공간에서 calibration pattern의 특징점 위치 (추후 2차원 사진 정보와 매칭됨) : X, Y, Z
objp[:, :2] = np.mgrid[0:wp, 0:hp].T.reshape(-1, 2)  # mgrid 사용하여 값 대입 (0, 0, 0) (1, 0, 0) ... (6, 7, 0)
objp[:, :2] *= length                                # 실제 크기 대입을 위해 length 곱함

objpoints = []   # 실제 세계에서의 3d 특징점 집합
imgpoints = []   # 촬영된 사진에서 2d 특징점 집합
findImages = []  # calibration pattern이 성공적으로 인식된 사진 번호 집합

for i in range(startImageNum, endImageNum):                         # 장 수에 비례한 accuaracy 확인할 수 있다.
    if i != 6 and i != 8:
        img = cv2.imread(os.path.join(directory, str(i) + imageExtension))      
        if img is None:
            print("failed to load image")
     
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)           
        ret, corners = cv2.findChessboardCorners(gray, (wp, hp), None)  # 사진에서 calibration pattern의 특징점 찾음
        img_shape = gray.shape[::-1]  # 사진의 가로 및 세로 화소 크기

        # 특징점이 사진에서 찾아진 경우
        if ret == True:
            print(f'{i}번째 이미지에서 calibration pattern 특징점 인식 성공')
            objpoints.append(objp)     # 집합에 3d정보 추가
            imgpoints.append(corners)  # 집합에 2d정보 추가
            findImages.append(i)       # 집합에 특징점 찾아진 사진 번호 추가
             
        # 특징점을 사진에서 찾지 못한 경우
        else:
            print(f'{i}번째 이미지에서 calibration pattern 특징점 인식 실패')
    
# camera calibration 수행
# M: intrinsic parameter, D: distortion parameter
# rvecs: camera와 calibration pattern의 좌표계 사이 rotation 정보
# rvecs: camera와 calibration pattern의 좌표계 사이 translation 정보
# rt: reprojection error,
rt, M, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None) # objpoints (대응시킬 pattern의 좌표), imgpoints (인식한 특징점 좌표), img_shape (img 크기),

print(f'Intrinsic matrix M:\n{M}')
np.savetxt(directory + "\Cal_intrinsic.txt", M, fmt='%.2f')  # intrinsic matrix 저장

# 외부 파라미터는 패턴 또는 카메라가 움직일 때 마다 변경, 내부 파라미터는 카메라 내부 특성으로 고정
W = np.full((3,4),0.0)
R = np.full((3,3),0.0)

for i, no in enumerate(findImages):
    rvec = rvecs[i]
    tvec = tvecs[i]
    
    cv2.Rodrigues(rvec,R)
    W[0:3,0:3] = R
    W[0:3,3:4] = tvec # np.reshape(tvec,(3,1))
    print(f'{no}th Extrinsic matrix W:\n{W}')
    np.savetxt(directory + "\Cal_extrinsic"+str(no)+".txt", W, fmt='%.2f')  # extrinsic matrix 저장
 
print(f'Distorsion coefficient:\n{D}')
print(f"Reprojection error: {rt}")

# After the camera calibration
for i in range(startImageNum, endImageNum):
    # Select an image for pose estimation
    pose_image_index = 0  # Replace with the index of the image you want to use for pose estimation
    pose_image = cv2.imread(os.path.join(directory, str(findImages[pose_image_index]) + imageExtension))
    pose_gray = cv2.cvtColor(pose_image, cv2.COLOR_BGR2GRAY)

    # Detect corners on the pose image
    ret, corners = cv2.findChessboardCorners(pose_gray, (wp, hp), None)

    # If corners are detected, estimate the object pose
    if ret:
        _, rvec, tvec = cv2.solvePnP(objpoints[pose_image_index], corners, M, D)
        R, _ = cv2.Rodrigues(rvec)

        print("Object Pose:")
        print("Rotation:")
        print(R)
        print("Translation:")
        print(tvec)
    else:
        print("Failed to detect corners on the pose image.")