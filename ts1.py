import stitch as st
import cv2 as cv


image_left = cv.imread("D:/pycharm/cv/dateset/18-1.jpg")
image_right = cv.imread("D:/pycharm/cv/dateset/18-2.jpg")

cv.imshow("left",image_left)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow("right",image_right)
cv.waitKey(0)
cv.destroyAllWindows()
a=st.Stitch()

a.match(image_left,image_right)