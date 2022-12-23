import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv



class Stitch:
    Stitch=None

    """获取匹配关键点与特征描述符"""
    def Detect_Feature_And_KeyPoints(self, image):
        detector = cv.SIFT_create()
        (Keypoints, descriptors) = detector.detectAndCompute(image, None)
        Keypoints = np.float32([i.pt for i in Keypoints])
        return (Keypoints, descriptors)

    """KNN匹配特征描述符"""
    def Descriptors_Match(self,descriptors_A,descriptors_B):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        knn_Matches = flann.knnMatch(descriptors_A, descriptors_B, 2)
        return knn_Matches


    """去除特征匹配点对列表中的非真实点对"""
    def Good_Match(self,knn_Matches,lowe_ratio=0.7):
        good = []
        for m in knn_Matches:
            if m[0].distance < lowe_ratio * m[1].distance:
                good.append((m[0].trainIdx,m[0].queryIdx))
        return good

    """提取视角变换矩阵M，与图像掩膜matchesMask"""
    def Good_Select(self,good,Keypoints_A,Keypoints_B,MIN_MATCH_COUNT=30):
        if len(good)>MIN_MATCH_COUNT:
            src_Points = np.float32([Keypoints_A[i] for (_,i) in good])
            dst_Points = np.float32([Keypoints_B[i] for (i,_) in good])
            M,mask=cv.findHomography(src_Points,dst_Points,cv.RANSAC,4.0)  #RANSAC减少误差
            matchesMask = mask.ravel().tolist()
            return M,matchesMask
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            return None,None


    """水平最优缝合线算法（基于动态规划算法的实现）"""
    def quilting(self,overlap1, overlap2, mid,RGB=True):
        print(overlap1.shape)
        print(overlap2.shape)
        E = np.full((overlap1.shape[0],overlap1.shape[1]),0)
        trace = np.zeros_like(E)
        if RGB==False:
            for i in range(E.shape[0]):
                for j in range(E.shape[1]):
                    E[i][j] = abs(int(overlap1[i][j]) - int(overlap2[i][j]))
        else:
            for i in range(E.shape[0]):
                for j in range(E.shape[1]):
                    E[i][j]=abs(int(overlap1[i][j][0])+int(overlap1[i][j][1])+int(overlap1[i][j][2])- int(overlap2[i][j][0])-int(overlap2[i][j][1])-int(overlap2[i][j][2]))
        a = np.ndarray(E.shape[1])
        for i in range(len(a)):
            a[i] = -1
        min_error = np.zeros(E.shape)
        for i in range(min_error.shape[0]):
            for j in range(min_error.shape[1]):
                min_error[i][j] = float('inf')
                trace[i][j] = -1
        min_error[mid][0] = E[mid][0]
        for j in range(min_error.shape[1] - 1):
            for i in range(min_error.shape[0]):
                if i == 0:
                    t = float('inf')
                    for k in (i, i + 1):
                        if min_error[k][j] < t:
                            t = min_error[k][j]
                            trace[i][j + 1] = k
                    min_error[i][j + 1] = t + E[i][j + 1]
                elif i == min_error.shape[0] - 1:
                    t = float('inf')
                    for k in (i - 1, i):
                        if min_error[k][j] < t:
                            t = min_error[k][j]
                            trace[i][j + 1] = k
                    min_error[i][j + 1] = t + E[i][j + 1]
                else:
                    t = float('inf')
                    for k in (i - 1, i, i + 1):
                        if min_error[k][j] < t:
                            t = min_error[k][j]
                            trace[i][j + 1] = k
                    min_error[i][j + 1] = t + E[i][j + 1]
        min = float('inf')
        position = -1
        for i in range(min_error.shape[0]):
            if min_error[i][min_error.shape[1] - 1] < min:
                min = min_error[i][min_error.shape[1] - 1]
                position = i
        a[overlap1.shape[1] - 1] = position
        for i in range(1, overlap1.shape[1]):
            a[overlap1.shape[1] - 1 - i] = trace[int(a[overlap1.shape[1] - i])][overlap1.shape[1] - i]
        return a.astype(int)

    """视角变换矩阵改良，保证图片变换结果完整性"""
    def M_fix(self,M,img1):
        x1 = np.dot(M, [0, 0, 1])
        y1 = np.dot(M, [0,img1.shape[0]-1, 1])
        x = np.dot(M, [img1.shape[1]-1,0, 1])
        y = np.dot(M, [img1.shape[1]-1, img1.shape[0]-1, 1])
        x_excursion = min(y1[0], x1[0])
        if x_excursion < 0:
            x_excursion = -x_excursion
        else:
            x_excursion = 0
        y_excursion = min(x1[1], x[1])
        if y_excursion < 0:
            y_excursion = -y_excursion
        else:
            y_excursion = 0
        move = np.float32(M)
        move[0][0] = move[0][0] + x_excursion * move[2][0]
        move[0][1] = move[0][1] + x_excursion * move[2][1]
        move[0][2] = move[0][2] + x_excursion * move[2][2]
        move[1][0] = move[1][0] + y_excursion * move[2][0]
        move[1][1] = move[1][1] + y_excursion * move[2][1]
        move[1][2] = move[1][2] + y_excursion * move[2][2]
        return move

    """以变换图为参照时,调用去图像黑边函数"""

    def change_size(self,image):
        img = cv.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
        b = cv.threshold(img, 15, 255, cv.THRESH_BINARY)  # 调整裁剪效果
        binary_image = b[1]  # 二值图--具有三通道
        binary_image = cv.cvtColor(binary_image, cv.COLOR_BGR2GRAY)

        indexes = np.where(binary_image == 255)  # 提取白色像素点的坐标

        left = min(indexes[0])  # 左边界
        right = max(indexes[0])  # 右边界
        width = right - left  # 宽度
        bottom = min(indexes[1])  # 底部
        top = max(indexes[1])  # 顶部
        height = top - bottom  # 高度

        pre1_picture = image[left:left + width, bottom:bottom + height]  # 图片截取
        return pre1_picture


    """图像拼接"""
    def getwarp_perspective(self,img1, img2, M,Anchor_Point):

        (x_anchor,y_anchor,x_anchor1,y_anchor1)=Anchor_Point
        img3 = img1.copy()
        img4 = img2.copy()
        Anchor=[x_anchor,y_anchor,1]
        Anchor=np.dot(M,Anchor)

        lt=[0,0,1]

        """x[0]:横坐标 x[1]:纵坐标"""
        lb=[0,img3.shape[0]-1,1]
        rt=[img3.shape[1]-1,0,1]
        rb=[img3.shape[1]-1,img3.shape[0]-1,1]

        lt = np.dot(M,lt)
        rt = np.dot(M, rt)
        lb = np.dot(M, lb)
        rb = np.dot(M, rb)
        Y_Anchor_Point=Anchor[1]/Anchor[2]
        X_Anchor_Point=Anchor[0]/Anchor[2]
        tran_y_max=max(rb[1]/rb[2],lb[1]/lb[2])
        tran_x_max=max(rb[0]/rb[2],rt[0]/rt[2])
        shape_y=max(Y_Anchor_Point,y_anchor1)+max(tran_y_max-Y_Anchor_Point,img4.shape[0]-y_anchor1)
        shape_x=max(X_Anchor_Point,x_anchor1)+max(tran_x_max-X_Anchor_Point,img4.shape[1]-x_anchor1)
        result=cv.warpPerspective(img1,M,(round(shape_x),round(shape_y)))


        if X_Anchor_Point>x_anchor1:
            left=round(X_Anchor_Point-x_anchor1)
        else:
            left=0
        if Y_Anchor_Point>y_anchor1:
            top=round(Y_Anchor_Point-y_anchor1)
        else:
            top=0

        result[top:top+img4.shape[0],left:left+img4.shape[1]]=img4


        return  result


    """测试用，画出匹配基准图"""
    def get_points(self,imageA,imageB):
        print(imageA.shape)
        print(imageB.shape)
        (hA, wA) = self.get_image_dimension(imageA)
        (hB, wB) = self.get_image_dimension(imageB)
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        return vis

    """获得图像二维长宽信息"""
    def get_image_dimension(self,image):
        (h,w) = image.shape[:2]
        return (h,w)



    """获取拼接图像特征匹配特征点矩形区域坐标"""
    """x_min.x_max,y_min,y_max为第一幅图片特征区域的横纵坐标最大，最小值"""
    def Get_Keypoint_range(self,kp1,kp2,good,matchmask):
        x_max = kp1[good[0][1]][0]
        x_min = kp1[good[0][1]][0]
        y_max = kp1[good[0][1]][1]
        y_min = kp1[good[0][1]][1]
        x_max1 = kp2[good[0][0]][0]
        x_min1 = kp2[good[0][0]][0]
        y_max1 = kp2[good[0][0]][1]
        y_min1 = kp2[good[0][0]][1]
        for i in range(len(good)):
            if matchmask[i]==1:
                if kp1[good[i][1]][0] > x_max:
                    x_max = kp1[good[i][1]][0]
                elif kp1[good[i][1]][0] < x_min:
                    x_min = kp1[good[i][1]][0]
                if kp1[good[i][1]][1] > y_max:
                    y_max = kp1[good[i][1]][1]
                elif kp1[good[i][1]][1] < y_min:
                    y_min = kp1[good[i][1]][1]
                if kp2[good[i][0]][0] > x_max1:
                    x_max1 = kp2[good[i][0]][0]
                elif kp2[good[i][0]][0] < x_min1:
                    x_min1 = kp2[good[i][0]][0]
                if kp2[good[i][0]][1] > y_max1:
                    y_max1 = kp2[good[i][0]][1]
                elif kp2[good[i][0]][1] < y_min1:
                    y_min1 = kp2[good[i][0]][1]
        return x_max,x_min,y_max,y_min,x_max1,x_min1,y_max1,y_min1


    """获取图像匹配锚点"""
    def Get_Anchor_Point(self,kp1,kp2,good,matchmask):
        for i in range(len(good)):
            if matchmask[i]==1:
                return kp1[good[i][1]][0],kp1[good[i][1]][1],kp2[good[i][0]][0],kp2[good[i][0]][1]


    """图像匹配函数"""
    def match(self,img1,img2,lowe_ratio=0.7):
        kp1, des1 = self.Detect_Feature_And_KeyPoints(img1)
        kp2, des2 = self.Detect_Feature_And_KeyPoints(img2)
        knn_matches=self.Descriptors_Match(des1,des2)
        good=self.Good_Match(knn_matches)
        M,matchesMask=self.Good_Select(good,kp1,kp2)
        if np.array(M).any():
            M_fix = self.M_fix(M, img1)
            Anchor_Point=self.Get_Anchor_Point(kp1,kp2,good,matchesMask)
            result=self.getwarp_perspective(img1,img2,M_fix,Anchor_Point)
            cv.imshow("result",result)
            cv.waitKey(0)
            cv.destroyAllWindows()

            result1=cv.cvtColor(result,cv.COLOR_BGR2RGB)
            self.panoramic=result
            cv.imwrite("orthographic perspective result.jpg", self.panoramic)
            return 1
        else:
            print("特征不足，匹配失败")
            return 0

