import cv2
import numpy as np
from imutils.perspective import four_point_transform

class CrossDetection:
    def __init__(self):
        # Inicializa o detector ORB
        self.orb = cv2.ORB_create()

        # Carrega a imagem da base para criar os descritores
        self.base_image = cv2.imread('/path/to/base_image.png', 0)
        self.keypoints_base, self.descriptors_base = self.orb.detectAndCompute(self.base_image, None)

    def apply_filters(self, img, parameters):
        p = parameters

        lower = p[0]
        upper = p[1]

        blur = (p[2][0], p[2][0])
        erode = p[2][1]
        dilate = p[2][2]

        lower_color = np.array(lower)
        upper_color = np.array(upper)

        if p[2][0] != 0:
            img = cv2.blur(img, blur)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_color, upper_color)
        img_mask = cv2.bitwise_and(img, img, mask=mask)

        erode_kernel = np.ones((erode, erode), np.float32)
        dilate_kernel = np.ones((dilate, dilate), np.float32)
        dilate = cv2.dilate(img_mask, dilate_kernel)
        erode = cv2.erode(dilate, erode_kernel)

        canny = cv2.Canny(erode, 200, 300)

        return canny

    def find_potentials(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shapes = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            
            if len(approx) == 4 and cv2.arcLength(contour, True) > 200:
                shapes.append(approx)
        
        return shapes

    def cluster(self, shapes):
        cluster = []
        LIM = 10
        i = 0
        while i < len(shapes):
            coord1 = shapes[i]
            cluster = [coord1]
            x_tot = coord1[0]
            y_tot = coord1[1]
            x_min = coord1[0] - LIM
            x_max = coord1[0] + LIM
            y_min = coord1[1] - LIM
            y_max = coord1[1] + LIM
            for j in range(len(shapes) - (i + 1)):
                coord2 = shapes[j + (i + 1)]
                if x_min <= coord2[0] <= x_max and y_min <= coord2[1] <= y_max:
                    cluster.append(coord2)
                    x_tot += coord2[0]
                    y_tot += coord2[1]
            if len(cluster) >= 3:
                return True
            i += 1
        return False

    def verify(self, shape, image):
        square = four_point_transform(image, shape.reshape(4, 2))
        contours, _ = cv2.findContours(square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shapes = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            
            if len(approx) < 20 and cv2.arcLength(contour, True) > 100:
                M = cv2.moments(contour)
                if M['m00'] != 0.0:
                    x = int(M['m10'] / M['m00'])
                    y = int(M['m01'] / M['m00'])
                    shapes.append((x, y))

        return self.cluster(shapes)

    def base_detection(self, img, fixed_bases, tolerance=1.0):
        # Parâmetros ajustados de calibração
        parameters = [[50, 50, 50], [255, 255, 255], [5, 5, 5]]

        img_filter = self.apply_filters(img, parameters)
        list_of_potentials = self.find_potentials(img_filter)

        result_fixed = []
        result_moving = []
        tol = 5
        for potential in list_of_potentials:
            if self.verify(potential, img_filter):
                M = cv2.moments(potential)
                if M['m00'] != 0.0:
                    x = int(M['m10'] / M['m00'])
                    y = int(M['m01'] / M['m00'])
                    new_cross = True
                    for point in result_fixed + result_moving:
                        if (point[0] - tol) <= x <= (point[0] + tol) and (point[1] - tol) <= y <= (point[1] + tol):
                            new_cross = False
                    if new_cross:
                        position = (x, y)
                        if self.is_near_fixed_base(position, fixed_bases, tolerance):
                            result_fixed.append(position)
                        else:
                            result_moving.append(position)

        # Detecção usando ORB
        keypoints_frame, descriptors_frame = self.orb.detectAndCompute(img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.descriptors_base, descriptors_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        for match in matches:
            frame_point = keypoints_frame[match.trainIdx].pt
            if self.is_near_fixed_base(frame_point, fixed_bases, tolerance):
                result_fixed.append(frame_point)
            else:
                result_moving.append(frame_point)

        return result_fixed, result_moving

    def is_near_fixed_base(self, base_position, fixed_bases, tolerance):
        for fixed_base in fixed_bases:
            distance = np.linalg.norm(np.array(base_position) - np.array(fixed_base))
            if distance <= tolerance:
                return True
        return False
