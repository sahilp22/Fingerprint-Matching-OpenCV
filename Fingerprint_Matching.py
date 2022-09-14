import os
import cv2

class detectMatch:
    sample = cv2.imread("C:\\Projects\\Fingerprint_Matching\\fingerprint_dataset\\SOCOFing\\Altered\\Altered-Hard\\150__M_Right_index_finger_Obl.BMP") 

    high_score = 0
    filename = None
    img = None
    kp1, kp2, mp = None, None, None

    counter = 0

    for file in [file for file in os.listdir("SOCOFing/Real")][:1000]:
        if counter % 10 == 0:
            print(counter)
            print(file)
        counter += 1

        fingerprint_img = cv2.imread("SOCOFing/Real/" + file)
        sift = cv2.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_img, None)
        
        matches = cv2.FlannBasedMatcher({'algorithm' : 1, 'trees': 10}, 
                                        {}).knnMatch(descriptors_1, descriptors_2, k=2) # Algorithm 1 is KD tree
                

        match_points = []

        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                match_points.append(p)
        
        keypoints = 0
        if len(keypoints_1) < len(keypoints_2):
            keypoints = len(keypoints_1)
        else:
            keypoints = len(keypoints_2)

        if len(match_points) / keypoints * 100 > high_score:
            high_score = len(match_points) / keypoints * 100 # Calculate the score
            filename = file
            img = fingerprint_img
            kp1, kp2, mp = keypoints_1, keypoints_2, match_points

    print("BEST MATCH: " + filename)
    print("SCORE: " + str(high_score))

    result = cv2.drawMatches(sample, kp1, img, kp2, mp, None, None)
    result = cv2.resize(result, None, fx=4, fy=4)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    match = detectMatch()
