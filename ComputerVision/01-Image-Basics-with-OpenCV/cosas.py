import cv2

img = cv2.imread("DATA/00-puppy.jpg")

w_scale = 0.5
h_scale = 0.5

img = cv2.resize(img, (0, 0), img, w_scale, h_scale)

while True:

    cv2.imshow("Puppy", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
