import cv2


class SIFTMatch:
    def __init__(self):
        img1 = cv2.imread("lena.png")
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        self.sift = cv2.SIFT_create()
        keypoints1, self.descriptors1 = self.sift.detectAndCompute(gray1, None)

    def matching(self, img):
        try:
            # グレースケール化
            gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 画像からSIFTを用いて特徴量を抽出
            keypoints2, descriptors2 = self.sift.detectAndCompute(gray2, None)

            # マッチャーを作成
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(self.descriptors1, descriptors2, k=2)

            # マッチングをフィルタリング
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            print(len(good_matches))
            if len(good_matches) > 100:
                return True
            else:
                return False
        except:
            return False
