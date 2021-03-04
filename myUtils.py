import numpy as np
import cv2
import dlib

facial_landmarks_idxs = {
    "jaw_line":range(0,17),
    "left_eyebrow":range(17,22),
    "right_eyebrow":range(22,27),
    "nose":range(27,36),
    "left_eye":range(36,42),
    "right_eye":range(42,48),
    "lips":range(48,61),
    "mouth_opening":range(61,68)
}

# imagem: imagem contendo o rosto
# predictor: preditor de formato do rosto (68 landmarks)
# face_loc: localização da rosto na imagem (padrao dlib.rectangle)
# retorna um numpy array contendo as marcações faciais (x,y)
def get_landmarks(image, predictor, face_loc=None):  
    # obs.: supondo que existe apenas um rosto por imagem e ocupa toda a imagem
    if face_loc == None:
        face_loc = dlib.rectangle(0, 0, image.shape[0], image.shape[1])
    landmarks = predictor(image, face_loc)
    
    #convertendo os pontos para numpy array
    pts = [(landmarks.part(n).x,landmarks.part(n).y) for n in range(68)]
    pts = np.array(pts, dtype='int')
    
    return pts

def eye_aspect_ratio(eye_points):
    eye_points = np.array(eye_points, dtype=float)
    
    V1 = np.linalg.norm(eye_points[1] - eye_points[5])
    V2 = np.linalg.norm(eye_points[2] - eye_points[4])
    H = np.linalg.norm(eye_points[3] - eye_points[0])
    
    r = (V1+V2)/(2.0*H)
    
    return r

def eyes_segmentation(image, landmarks):
    eye_l_pts = landmarks[facial_landmarks_idxs['left_eye']]    
    eye_r_pts = landmarks[facial_landmarks_idxs['right_eye']]
    
    mask = np.zeros_like(image)
    mask = cv2.drawContours(mask, [eye_l_pts], 0, 255, cv2.FILLED)
    mask = cv2.drawContours(mask, [eye_r_pts], 0, 255, cv2.FILLED)
    
    segmented = cv2.bitwise_and(image, mask)
    
    return segmented, mask

def draw_landmarks(image, landmarks):
    img_out = image.copy()
    
    for landmark in landmarks:
        img_out = cv2.circle(img_out, tuple(landmark), 1 , 255, cv2.FILLED)    
    
    return img_out


################# METRICS #################
def confusion_matrix(y_true, y_pred, n_classes=2):
    
    assert(len(y_true) == len(y_pred))
    
    cm = np.zeros([n_classes, n_classes], dtype='int')
    
    y_true = y_true.astype('int')
    y_pred = y_pred.astype('int')
    
    
    for i in range(len(y_true)):
        real = y_true[i]
        pred = y_pred[i]
        cm[real,pred] += 1
    
    return cm

def compute_metrics_from_cm(cm):
    metrics = {}
    
    TP, FN, FP, TN = cm.flatten()
    
    metrics["accuracy"] =  (TP + TN)/cm.sum()
    metrics["precision"]=  TP / (TP + FP)
    metrics["recall"]   =  TP / (TP + FN)
    metrics["f1_score"] =  2.0*TP/(2.0*TP + FP + FN)
    
    return metrics

############### FEATURES EXTRACTORS ###############

# EAR
def feature_extrator_EAR(image, landmarks):
    #pega apenas os 6 pontos que representam a regiao de interesse
    #pontos do olho esquerdo
    left_eye_pts  = landmarks[facial_landmarks_idxs['left_eye'],:]
    #pontos do olho direito
    right_eye_pts = landmarks[facial_landmarks_idxs['right_eye'],:]
    
    ear_l = eye_aspect_ratio(left_eye_pts)
    ear_r = eye_aspect_ratio(right_eye_pts)
    
    return [ear_l, ear_r]

# HIST
def feature_extrator_HIST(image, landmarks):
    #segmenta a imagem
    segmented, _ = eyes_segmentation(image, landmarks)
    
    #quantidade de bins testado empiricamente, 8~10 aparentou ser um bom número
    bins = 8
    ranges= [1,255]#começa em 1 para ignorar os zeros da imagem (imagem vira segmentada)

    hist = cv2.calcHist([segmented], channels=[0], mask = None, histSize=[bins], ranges=ranges)
    #normaliza os valores do histograma para [0,1]
    cv2.normalize(hist,hist)
    
    return hist.flatten().tolist()

# EAR + HIST
def feature_extrator_EAR_HIST(image, landmarks):
    features = feature_extrator_EAR(image, landmarks) + feature_extrator_HIST(image,landmarks)
    return features