import numpy as np
import cv2
import transforms3d as tf

def getHM(r,t):

    if(np.shape(r)==(3,1)):
        rmat,_ = cv2.Rodrigues(r)
        hm  = np.concatenate((rmat, t), axis=1)
        HM = np.concatenate((hm,[[0,0,0,1]]),axis=0)
    else:
        hm  = np.concatenate((r, t), axis=1)
        HM = np.concatenate((hm,[[0,0,0,1]]),axis=0)

    return HM

def getPose(HM,unit):
    rmat = HM[0:3,0:3]
    tvec = HM[0:3,3]
    Position = [0,0,0]

    Angles = tf.euler.mat2euler(rmat)
    Angles = np.array(Angles) *57.2958  #sxyz
    # Cube position from translation vector
    if unit == "mm":
        Position[0] = tvec[0]  * 1000  
        Position[1]=  tvec[1]  * 1000   
        Position[2]=  tvec[2]  * 1000
    elif unit == "cm":
        Position[0] = tvec[0]  * 100  
        Position[1]=  tvec[1]  * 100   
        Position[2]=  tvec[2]  * 100
    Quaternion = tf.quaternions.mat2quat(rmat)  #wxyz
    Position = np.array(Position).round(3)
    Quaternion = np.array(Quaternion).round(3)
    Angles = Angles.round(0)
    return Position,Quaternion, Angles


def pose_estimation(frame, aruco_dict,size, matrix_coefficients, distortion_coefficients,Translations,Rotations):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dict,parameters)
    
    corners, ids, rejected = detector.detectMarkers(gray)

    objPoints = [[-size/2,size/2,0],[size/2,size/2,0],[size/2,-size/2,0],[-size/2,-size/2,0]]
    op = np.array(objPoints).reshape(4,1,3)
    shape = np.shape(corners)

    #no markers detected
    if shape[0] == 0:
        return frame,False,None
    corners = np.array(corners)

   
    # Choose the lowest id ArUco
    minId = min(ids)[0]
    minIndex = np.argmin(ids)
    if minId > 5 :
        return frame,False,None
    
    #vectors from camera to marker Tc_c2m
    _, rvec, tvec = cv2.solvePnP(op,corners[minIndex], matrix_coefficients,distortion_coefficients,flags=cv2.SOLVEPNP_IPPE_SQUARE)
    frame = cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.03,2)

    #Cam2Marker
    HM_cam2marker = getHM(rvec,tvec)

    #Hand-Eye Calibration Result Base to Camera
    HeC_b2c = np.load("/home/chichu/Desktop/PoseEstimation-main/real_robot/hec_camera_poses/Tb_b2c_20230726_CSE4144_front.npy")
    
    #base to marker
    HM_base2marker = np.matmul(HeC_b2c,HM_cam2marker)
    t1 = HM_base2marker

    
    zero = np.array([0,0,0],dtype=float)
    zero = zero.reshape(3,1)
    

    # Marker to Marker0 (rotation only)
    HM_marker2marker0 =  getHM(Rotations[minId],zero)
    
    #base to Marker0
    HM_base2marker0 = np.matmul(HM_base2marker,HM_marker2marker0)

    #Marker0 to Cube
    translation = Translations[minId]
    translation = translation.reshape(3,1)
    HM_marker02cube = getHM(zero,translation)
    
    #base to Cube
    HM_base2cube = np.matmul(HM_base2marker0,HM_marker02cube)

    return frame, True, HM_base2cube
