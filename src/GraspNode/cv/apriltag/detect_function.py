import pupil_apriltags as apriltag     # 在 windows 下引入该库
import cv2

# 输出包含图片corner坐标和center坐标
def position(img):
    gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    detector = apriltag.Detector()
    result = detector.detect(gray)
    return result

