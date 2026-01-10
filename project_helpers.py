import numpy as np

def get_dataset_info(dataset):
    """
    Datasets 1-5 are relatively easy, since they have little lens distortion and
    no dominant scene plane. You should get reasonably good reconstructions for
    these datasets using RANSAC to estimate E only (CE2 in assignment 4).

    Datasets 6-9 are much more difficult, since the scene is dominated by a near
    planar structure. Also the lens distortion is larger for the used camera,
    hence feel free to adjust the pixel_threshold if necessary.
    """
    if dataset == 1:
        img_names = ["data/1/kronan1.JPG", "data/1/kronan2.JPG"]
        im_width = 1936
        im_height = 1296
        focal_length_35mm = 45.0 # from the EXIF data
        init_pair = [0, 1]
        pixel_threshold = 1.0
    elif dataset == 2:
        # Corner of a courtyard
        img_names = ["data/2/DSC_0025.JPG", "data/2/DSC_0026.JPG", "data/2/DSC_0027.JPG", "data/2/DSC_0028.JPG", "data/2/DSC_0029.JPG", "data/2/DSC_0030.JPG", "data/2/DSC_0031.JPG", "data/2/DSC_0032.JPG", "data/2/DSC_0033.JPG"]
        im_width = 1936
        im_height = 1296
        focal_length_35mm = 43.0 # from the EXIF data
        init_pair = [0, 8]
        pixel_threshold = 1.0
    elif dataset == 3:
        # Smaller gate of a cathetral
        img_names = ["data/3/DSC_0001.JPG", "data/3/DSC_0002.JPG", "data/3/DSC_0003.JPG", "data/3/DSC_0004.JPG", "data/3/DSC_0005.JPG", "data/3/DSC_0006.JPG", "data/3/DSC_0007.JPG", "data/3/DSC_0008.JPG", "data/3/DSC_0009.JPG", "data/3/DSC_0010.JPG", "data/3/DSC_0011.JPG", "data/3/DSC_0012.JPG"]
        im_width = 1936
        im_height = 1296
        focal_length_35mm = 43.0 # from the EXIF data
        init_pair = [4, 7]
        pixel_threshold = 1.0
    elif dataset == 4:
        # Fountain
        img_names = ["data/4/DSC_0480.JPG", "data/4/DSC_0481.JPG", "data/4/DSC_0482.JPG", "data/4/DSC_0483.JPG", "data/4/DSC_0484.JPG", "data/4/DSC_0485.JPG", "data/4/DSC_0486.JPG", "data/4/DSC_0487.JPG", "data/4/DSC_0488.JPG", "data/4/DSC_0489.JPG", "data/4/DSC_0490.JPG", "data/4/DSC_0491.JPG", "data/4/DSC_0492.JPG", "data/4/DSC_0493.JPG"]
        im_width = 1936
        im_height = 1296
        focal_length_35mm = 43.0 # from the EXIF data
        init_pair = [4, 9]
        pixel_threshold = 1.0
    elif dataset == 5:
        # Golden statue
        img_names = ["data/5/DSC_0336.JPG", "data/5/DSC_0337.JPG", "data/5/DSC_0338.JPG", "data/5/DSC_0339.JPG", "data/5/DSC_0340.JPG", "data/5/DSC_0341.JPG", "data/5/DSC_0342.JPG", "data/5/DSC_0343.JPG", "data/5/DSC_0344.JPG", "data/5/DSC_0345.JPG"]
        im_width = 1936
       
        im_height = 1296
        focal_length_35mm = 45.0 # from the EXIF data
        init_pair = [2, 6]
        pixel_threshold = 1.0
    elif dataset == 6:
        # Detail of the Landhaus in Graz.
        img_names = ["data/6/DSCN2115.JPG", "data/6/DSCN2116.JPG", "data/6/DSCN2117.JPG", "data/6/DSCN2118.JPG", "data/6/DSCN2119.JPG", "data/6/DSCN2120.JPG", "data/6/DSCN2121.JPG", "data/6/DSCN2122.JPG"]
        im_width = 2272
       
        im_height = 1704
        focal_length_35mm = 38.0 # from the EXIF data
        init_pair = [1, 3]
        pixel_threshold = 1.0
    elif dataset == 7:
        # Building in Heidelberg.
        img_names = ["data/7/DSCN7409.JPG", "data/7/DSCN7410.JPG", "data/7/DSCN7411.JPG", "data/7/DSCN7412.JPG", "data/7/DSCN7413.JPG", "data/7/DSCN7414.JPG", "data/7/DSCN7415.JPG"]
        im_width = 2272
        im_height = 1704
        focal_length_35mm = 38.0 # from the EXIF data
        init_pair = [0, 6]
        pixel_threshold = 1.0
    elif dataset == 8:
        # Relief
        img_names = ["data/8/DSCN5540.JPG", "data/8/DSCN5541.JPG", "data/8/DSCN5542.JPG", "data/8/DSCN5543.JPG", "data/8/DSCN5544.JPG", "data/8/DSCN5545.JPG", "data/8/DSCN5546.JPG", "data/8/DSCN5547.JPG", "data/8/DSCN5548.JPG", "data/8/DSCN5549.JPG", "data/8/DSCN5550.JPG"]
        im_width = 2272
        im_height = 1704
        focal_length_35mm = 38.0 # from the EXIF data
        init_pair = [3, 6]
        pixel_threshold = 1.0
    elif dataset == 9:
        # Triceratops model on a poster.
        img_names = ["data/9/DSCN5184.JPG", "data/9/DSCN5185.JPG", "data/9/DSCN5186.JPG", "data/9/DSCN5187.JPG", "data/9/DSCN5188.JPG", "data/9/DSCN5189.JPG", "data/9/DSCN5191.JPG", "data/9/DSCN5192.JPG", "data/9/DSCN5193.JPG"]
        im_width = 2272
        im_height = 1704
        focal_length_35mm = 38.0 # from the EXIF data
        init_pair = [3, 5]
        pixel_threshold = 1.0
    elif dataset == 10:
        # Add your optional datasets...
        pass

    focal_length = max(im_width, im_height) * focal_length_35mm / 35.0
    K = np.array([[focal_length, 0, im_width/2], [0, focal_length, im_height/2], [0, 0, 1]])
    return K, img_names, init_pair, pixel_threshold

def correct_H_sign(H, x1, x2):
    N = x1.shape[1]
    if x1.shape[0] != 3:
        x1 = np.vstack([x1, np.ones_like(x1[:1,:])])
    if x2.shape[0] != 3:
        x2 = np.vstack([x2, np.ones_like(x2[:1,:])])

    positives = sum((sum(x2 * (H @ x1), 0)) > 0)
    if positives < N/2:
        H *= -1
    
    return H

def homography_to_RT(H):
    
    def unitize(a,b):
        denom = 1.0 / (a**2+b**2)**(0.5)
        ra = a * denom
        rb = b * denom
        return ra, rb

    [U,S,Vt] = np.linalg.svd(H)
    s1 = S[0] / S[1]
    s3 = S[2] / S[1]
    a1 = (1 - s3**2)**(0.5)
    b1 = (s1**2 - 1)**(0.5)
    [a,b] = unitize(a1, b1)
    [c,d] = unitize(1+s1*s3, a1*b1 )
    [e,f] = unitize(-b/s1, -a/s3 )
    v1 = Vt.T[:,0]
    v3 = Vt.T[:,2]
    n1 = b * v1 - a * v3
    n2 = b * v1 + a * v3
    R1 = U @ np.array([[c,0,d], [0,1,0], [-d,0,c]]) @ Vt
    R2 = U @ np.array([[c,0,-d], [0,1,0], [d,0,c]]) @ Vt
    t1 = (e * v1 + f * v3).reshape(-1,1)
    t2 = (e * v1 - f * v3).reshape(-1,1)
  
    if n1[2] < 0:
        t1 = -t1
        n1 = -n1

    if n2[2] < 0:
        t2 = -t2
        n2 = -n2

    t1 = R1 @ t1
    t2 = R2 @ t2

    RT = np.zeros((2,3,4))
    RT[0] = np.hstack([R1, t1])
    RT[1] = np.hstack([R2, t2])
    return RT