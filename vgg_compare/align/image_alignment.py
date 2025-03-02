import cv2
import numpy as np
from scipy.optimize import minimize
from skimage.metrics import peak_signal_noise_ratio as psnr


def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)


def otsu_threshold(image):

    # 計算影像的直方圖
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])

    # 總像素數量
    total = image.shape[0] * image.shape[1]

    # 計算直方圖的累積和以及累積均值
    sumB, sum1, q1, q2, varMax, threshold = 0, np.dot(np.arange(256), hist), 0, 0, 0, 0

    for i in range(256):
        q1 += hist[i]
        if q1 == 0:
            continue
        q2 = total - q1
        if q2 == 0:
            break

        sumB += i * hist[i]
        m1 = sumB / q1
        m2 = (sum1 - sumB) / q2

        # 計算類間方差
        varBetween = q1 * q2 * (m1 - m2) ** 2

        # 最大化類間方差來找到最佳閾值
        if varBetween > varMax:
            varMax = varBetween
            threshold = i

    return threshold


def estimate_initial_transform(img1, img2):
    # 使用ORB特征点检测
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 检查特征点和描述符是否成功提取
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print("Feature extraction failed for one or both images.")
        return [0, 0, 0, 1, 0, 0]  # 默认值包括shear

    # 使用暴力匹配器进行特征点匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 获取匹配点
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 使用RANSAC计算初步变换矩阵
    M, _ = cv2.estimateAffine2D(src_pts, dst_pts)

    if M is not None:
        # 提取旋转角度、平移量、缩放比例、和shear参数
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
        scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
        tx, ty = M[0, 2], M[1, 2]

        return [angle, tx, ty, scale, 0, 0]
    else:
        return [0, 0, 0, 1, 0, 0]  # 默认值包括shear


def loss_function(params, img1, img2):

    angle, tx, ty, scale, shear_x, shear_y = (
        params  # 旋转角度, X方向平移, Y方向平移, 缩放比例, shear
    )

    # 生成 shear 矩阵
    shear_matrix = np.array(
        [[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]]
    )  # 将 shear 矩阵扩展为3x3矩阵

    # 生成旋转矩阵并扩展为3x3矩阵
    M_rot = cv2.getRotationMatrix2D((img2.shape[1] / 2, img2.shape[0] / 2), angle, 1)
    M_rot = np.vstack([M_rot, [0, 0, 1]])

    # 生成缩放矩阵
    scale_matrix = np.array([[scale, 0, tx], [0, scale, ty], [0, 0, 1]])

    # 生成平移矩阵
    # translation_matrix = np.array([[1, 0, tx],
    #                  [0, 1, ty],
    #                  [0, 0, 1]])

    # 将所有变换矩阵按顺序相乘

    # shear -> rotate -> scale+translate
    M_combined = np.dot(scale_matrix, np.dot(M_rot, shear_matrix))

    # 对影像进行shear、旋转、缩放和平移变换
    transformed_img2 = cv2.warpAffine(
        img2,
        M_combined[:2],
        (img2.shape[1], img2.shape[0]),
        borderValue=(255, 255, 255),
    )

    similarity = psnr(img1, transformed_img2)

    # 返回 -相似性 作为损失
    return -similarity


def alignment(image1, image2):

    image1 = resize_image(image1)
    image2 = resize_image(image2)
    # 初始化旋转角度、平移量、缩放比例和shear
    initial_params = estimate_initial_transform(
        image1, image2
    )  # 初始角度, X平移, Y平移, 缩放比例, shear

    # 使用 scipy.optimize.minimize 进行优化
    result = minimize(
        loss_function, initial_params, args=(image1, image2), method="Nelder-Mead"
    )

    # 最优化参数
    (
        optimal_angle,
        optimal_tx,
        optimal_ty,
        optimal_scale,
        optimal_shear_x,
        optimal_shear_y,
    ) = result.x

    # 生成优化后的变换矩阵
    M_rot_scale = cv2.getRotationMatrix2D(
        (image2.shape[1] / 2, image2.shape[0] / 2), optimal_angle, optimal_scale
    )

    # 构造 shearing 矩阵 (这里假设 shearing 只发生在 x 方向和 y 方向)
    shear_matrix = np.array(
        [[1, optimal_shear_x, 0], [optimal_shear_y, 1, 0]], dtype=np.float32
    )

    # 将旋转矩阵与 shearing 矩阵相乘
    M = np.dot(shear_matrix, np.vstack([M_rot_scale, [0, 0, 1]]))[:2, :]

    # 添加平移量
    M[0, 2] += optimal_tx
    M[1, 2] += optimal_ty

    # 应用最终变换矩阵
    aligned_image = cv2.warpAffine(
        image2, M, (image2.shape[1], image2.shape[0]), borderValue=(255, 255, 255)
    )

    # Otsu thresholding
    k_threshold = otsu_threshold(aligned_image)
    _, aligned_image = cv2.threshold(aligned_image, k_threshold, 255, cv2.THRESH_BINARY)

    # print(f'Optimal rotation angle: {optimal_angle:.4f} degrees')
    # print(f'Optimal translation: ({optimal_tx:.4f}, {optimal_ty:.4f})')
    # print(f'Optimal scale: {optimal_scale:.4f}')
    # print(f'Optimal shear: ({optimal_shear_x:.4f}, {optimal_shear_y:.4f})')
    # print(f"before alignment PSNR : {psnr(image1, image2):.4f}")
    # print(f"after alignment PSNR : {psnr(image1, aligned_image):.4f}")

    if psnr(image1, aligned_image) < psnr(image1, image2):
        return image1, image2, psnr(image1, image2)
    else:
        return image1, aligned_image, psnr(image1, aligned_image)


def compare_images(image1, image2):

    image1_ref, image2_aligned, accuracy_1 = alignment(image1, image2)
    image2_ref, image1_aligned, accuracy_2 = alignment(image2, image1)

    if accuracy_1 > accuracy_2:
        return image1_ref, image2_aligned
    else:
        return image2_ref, image1_aligned


def compare_images_with_check(image1, image2):
    image1_ref, image2_aligned, accuracy_1 = alignment(image1, image2)
    image2_ref, image1_aligned, accuracy_2 = alignment(image2, image1)

    if accuracy_1 > accuracy_2:
        return image1_ref, image2_aligned, True
    else:
        return image2_ref, image1_aligned, False
