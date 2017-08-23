import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Camera calibration parameters
# Camera matrix
c_mtx = np.array(
	[[1.15396091e+03, 0.00000000e+00, 6.69706056e+02],
 	[0.00000000e+00, 1.14802495e+03, 3.85655654e+02],
 	[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
 	)
# Distortion coefficients
dist = np.array(
	[[ -2.41016964e-01, -5.30788794e-02, -1.15812035e-03,
	-1.28281652e-04, 2.67259026e-02]]
    )


# Helper functions
def bgr2rgb(image):
    """Convert from BGR to RGB."""
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pltimshow(img, figsize=None):
	"""Show image."""

	if figsize is not None:
		plt.figure(figsize=figsize)
	plt.imshow(img)
	plt.show()
	plt.close()



# define a function to undistort the raw images of cameras
def undistort(raw, c_mtx, dist):
    """Undistort raw camera images.
    
    This function uses the camera matrix (c_mtx), distortion
    coefficients (dist) to undistort raw camera images.
    
    Returns BGR images!
    
    Args:
        raw (ndarray): The image taken by the cameraof which no 
            distortion correction applied. Image should be in `BGR` 
            format, read with `cv2.imread`.
            
        c_mtx: Camera calibration matrix, can be obtained using the
            `cv2.calibrateCamera` module.
        
        dist: Distortion coefficients, can be obtained using the
            `cv2.calibrateCamera` module.
    """
    
    # get undistorted destination image
    undist = cv2.undistort(raw, c_mtx, dist, None, c_mtx)
    
    return undist


# define a color thresholding function
def color_thresholding(
    img, ch_type='rgb', 
    binary=True, plot=False, 
    thr=(220, 255), save_path=None):
    """Apply color thresholding.
    
    Arg:
        img (numpy array): numpy image array, should be in `RGB`
            color space, NOT in `BGR`.
            
        ch_type (str): can be 'rgb', 'hls', 'hsv', 'yuv', 'ycrcb',
            'lab', 'luv'.
            
        binary (bool): If `True` then show and returns binary
            images. If not, returns original images in defined 
            color spaces.
            
        plot: If `True`, shows images.
        
        thr: min, max value for threasholding.
        
        save_path: if defines, saves figures.
    """
    # get channels
    if ch_type is 'hls':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif ch_type is 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif ch_type is 'yuv':    
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif ch_type is 'ycrcb':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif ch_type is 'lab':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    elif ch_type is 'luv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    
    img_ch1 = img[:,:,0]
    img_ch2 = img[:,:,1]
    img_ch3 = img[:,:,2]

    # apply thresholding
    bin_ch1 = np.zeros_like(img_ch1)
    bin_ch2 = np.zeros_like(img_ch2)
    bin_ch3 = np.zeros_like(img_ch3)

    bin_ch1[(img_ch1 > thr[0]) & (img_ch1 <= thr[1])] = 1
    bin_ch2[(img_ch2 > thr[0]) & (img_ch2 <= thr[1])] = 1
    bin_ch3[(img_ch3 > thr[0]) & (img_ch3 <= thr[1])] = 1
    
    if binary:
        imrep_ch1 = bin_ch1
        imrep_ch2 = bin_ch2
        imrep_ch3 = bin_ch3
    else:
        imrep_ch1 = img_ch1
        imrep_ch2 = img_ch2
        imrep_ch3 = img_ch3
    if plot:
        n_rows = 2
        n_cols = 3
        fig, axarr = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 5, n_rows * 3),
            subplot_kw={'xticks': [], 'yticks': []})
        axarr[0, 0].imshow(img_ch1, cmap='gray')
        axarr[0, 0].set_title('original ' + ch_type + ' space: ch1')
        axarr[0, 1].imshow(img_ch2, cmap='gray')
        axarr[0, 1].set_title('original ' + ch_type + ' space: ch2')
        axarr[0, 2].imshow(img_ch3, cmap='gray')
        axarr[0, 2].set_title('original ' + ch_type + ' space: ch3')
        
        axarr[1, 0].imshow(imrep_ch1, cmap='gray')
        axarr[1, 0].set_title('binary ' + ch_type + ' space: ch1')
        axarr[1, 1].imshow(imrep_ch2, cmap='gray')
        axarr[1, 1].set_title('binary ' + ch_type + ' space: ch2')
        axarr[1, 2].imshow(imrep_ch3, cmap='gray')
        axarr[1, 2].set_title('binary ' + ch_type + ' space: ch3')
        plt.show()
        plt.close()
    
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.imsave(save_path + '/' + ch_type + '_ch1.png', imrep_ch1, cmap='gray')
        plt.imsave(save_path + '/' + ch_type + '_ch2.png', imrep_ch2, cmap='gray')
        plt.imsave(save_path + '/' + ch_type + '_ch3.png', imrep_ch3, cmap='gray')
    
    return imrep_ch1, imrep_ch2, imrep_ch3


def explore_img_color_thr(img, binary=False, save_path=None):
    """Explore image for various color thresholdings.
     
    Args:
        img1 (numpy array): numpy image array, should be in `RGB` color
            space, NOT in `BGR`.
        
        binary (bool): If `True`, returns binary representations.
            If `False`, returns original images in defined
            color spaces.
        
        save_path (str): if not None, figures are saved.
    """
    
    ch_types = ['rgb', 'hls', 'hsv', 'yuv', 'ycrcb', 'lab', 'luv']
    
    dst_images = {}
    dst_images['img1'] = {}
    for ch_type in ch_types:
        dst_images['img1'][ch_type] =\
            color_thresholding(img, ch_type=ch_type, binary=binary, save_path=save_path)
        
    n_cols = 3
    n_rows = len(ch_types)
    
    fig, axarr = plt.subplots(
        n_rows, n_cols, figsize=(15, 20),
        subplot_kw={'xticks': [], 'yticks': []})
    
    k = 0
    for ch_type in ch_types:
        axarr[k, 0].imshow(dst_images['img1'][ch_type][0], cmap='gray')
        axarr[k, 0].set_title(ch_type + ' space: ch1')
        
        axarr[k, 1].imshow(dst_images['img1'][ch_type][1], cmap='gray')
        axarr[k, 1].set_title(ch_type + ' space: ch2')
        
        axarr[k, 2].imshow(dst_images['img1'][ch_type][2], cmap='gray')
        axarr[k, 2].set_title(ch_type + ' space: ch3')
        k += 1

    plt.show()
    plt.close()

# define a function to compare two different images
def compare_2img_color_thr(img1, img2, binary=False, save_path=[None, None]):
    """Compares two images for various color thresholdings.
     
    Args:
        img1 (numpy array): numpy image array, should be in `RGB` color
            space, NOT in `BGR`.
        
        img2 (numpy array): numpy image array, should be in `RGB` color
            space, NOT in `BGR`.
        
        binary (bool): If `True`, returns binary representations.
            If `False`, returns original images in defined
            color spaces.
        
        save_path (str): if not None, figures are saved.
    """
    
    ch_types = ['rgb', 'hls', 'hsv', 'yuv', 'ycrcb', 'lab', 'luv']
    
    dst_images = {}
    dst_images['img1'] = {}
    dst_images['img2'] = {}
    for ch_type in ch_types:
        dst_images['img1'][ch_type] =\
            color_thresholding(img1, ch_type=ch_type, binary=binary, save_path=save_path[0])
        dst_images['img2'][ch_type] =\
            color_thresholding(img2, ch_type=ch_type, binary=binary, save_path=save_path[1])
        
    n_cols = 3 * 2
    n_rows = len(ch_types)
    
    fig, axarr = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2),
        subplot_kw={'xticks': [], 'yticks': []})
    
    k = 0
    for ch_type in ch_types:
        axarr[k, 0].imshow(dst_images['img1'][ch_type][0], cmap='gray')
        axarr[k, 0].set_title(ch_type + ' space: ch1')
        
        axarr[k, 1].imshow(dst_images['img1'][ch_type][1], cmap='gray')
        axarr[k, 1].set_title(ch_type + ' space: ch2')
        
        axarr[k, 2].imshow(dst_images['img1'][ch_type][2], cmap='gray')
        axarr[k, 2].set_title(ch_type + ' space: ch3')
        
        axarr[k, 3].imshow(dst_images['img2'][ch_type][0], cmap='gray')
        axarr[k, 3].set_title(ch_type + ' space: ch1')
        
        axarr[k, 4].imshow(dst_images['img2'][ch_type][1], cmap='gray')
        axarr[k, 4].set_title(ch_type + ' space: ch2')
        
        axarr[k, 5].imshow(dst_images['img2'][ch_type][2], cmap='gray')
        axarr[k, 5].set_title(ch_type + ' space: ch3')
        
        k += 1

    plt.show()
    plt.close()




# write a masking function for defining a region of interest
def region_of_interest(img, points):
    """Mask the region of interest
    
    Only keeps the region of the image defined by the polygon
    formed from `points`. The rest of the image is set to black.
    
    Args:
        img: numpy array representing the image.
        points: verticies, example:
            [[(x1, y1), (x2, y2), (x4, y4), (x3, y3)]]
    """
    # define empty binary mask
    mask = np.zeros_like(img)
    
    # define a mask color to ignore the masking area
    # if there are more than one color channels consider
    # the shape of ignoring mask color
    if len(img.shape) > 2:
        n_chs = img.shape[2]
        ignore_mask_color = (255,) * n_chs
    else:
        ignore_mask_color = 255
    
    # define unmasking region 
    cv2.fillPoly(mask, points, ignore_mask_color)
    
    # mask the image
    masked_img = cv2.bitwise_and(img, mask)
    
    return masked_img




# define a perspective transformation function
def birds_eye_transform(img, points, offsetx):
    """Transforms the viewpoint to a bird's-eye view.
    
    Applies a perspective transformation. Returns
    the inverse matrix and the warped destination
    image.
    
    Args:
        img: A numpy image array.
        points: A list of four points to be flattened.
            Example: points = [[x1,y1], [x2,y2], [x4,y4], [x3,y3]].
        offsetx: offset value for x-axis.
    """
    
    img_size = img[:,:,0].shape[::-1]
    
    # get the region of interest
    img = region_of_interest(img, np.array([points]))
    
    src = np.float32(
        [
            points[0],
            points[1],
            points[2],
            points[3],
        ])

    pt1 = [offsetx, 0]
    pt2 = [img_size[0] - offsetx, 0]
    pt3 = [img_size[0] - offsetx, img_size[1]]
    pt4 = [offsetx, img_size[1]]
    dst = np.float32([pt1, pt2, pt3, pt4])
    
    mtx = cv2.getPerspectiveTransform(src, dst)
    invmtx = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(img, mtx, img_size)
    
    return invmtx, warped




# Define a function to combine color thresholding
def combined_color_thresholding(
    img, thr_rgb=(230,255), thr_hsv=(230,255), thr_luv=(157,255)):
    """Combines color thresholding on different channels
    
    Returns a combined binary mask.
    
    Args:
        img: Numpy image array, should be in `RGB` color
            space.
        
        thr_rgb: min and max thresholding values for RGB color
            space.
        
        thr_hsv: min and max thresholding values for HSV color
            space.
        
        thr_luv: min and max thresholding values for LUV color
            space.
    """
    
    bin_rgb_ch1, bin_rgb_ch2, bin_rgb_ch3 =\
        color_thresholding(img, ch_type='rgb', thr=thr_rgb)
        
    bin_hsv_ch1, bin_hsv_ch2, bin_hsv_ch3 =\
    color_thresholding(img, ch_type='hsv', thr=thr_hsv)
    
    bin_luv_ch1, bin_luv_ch2, bin_luv_ch3 =\
    color_thresholding(img, ch_type='luv', thr=thr_luv)
    
    binary = np.zeros_like(bin_rgb_ch1)
    binary[
        (bin_rgb_ch1 == 1)
        | (bin_hsv_ch3 == 1)
        | (bin_luv_ch3 == 1)
        ] = 1
    
    return binary




def search_lane(warped_list, n_stepsy=9, n_stepsx=2, std_max=5., min_samp=50):
    """Search binary images to detect lane lines.
    
    Takes the top-view (warped) image as input, applies color
    thresholding to get a binary representation. Searches binary 
    image with sliding windows and detects lane pixels. In the 
    sliding window, if the standart deviation value for the white 
    pixels is lower than a threshold, detects as line pixels and 
    appends to result.
    
    returns an output image and the fitting parameters.
    
    Args:
        warped_list (list): list of numpy arrays representing the two top-view road images.
            Images should be in RGB format. One image is original, second one is histogram
            equalization is applied for severe shadow conditions.
        
        n_stepsy (int): number of slides on y-axis.
        
        n_stepsx (int): number of slides on x-axis (n_stepsx >= 2).
        
        std_max (float): Float to represent the maximum value of
            standart deviation.
        
        min_samp (int): minimum number of white pixels to considering
            to check if lane or not.
    """
    
    def stdx(binary_window, starty=0, startx=0):
        """Computes mean and standart deviation of a binary window.
    
        Computes and returns the standart deviation value of lane pixels
        in the binary search window, along x-axis. Also returns number of 
        samples (number of white pixels in this case) and the lists of the
        absolute position of white pixels (absolute x and y coordinates of 
        lane pixels) in the whole binary image.
        
        Args:
            binary_window: An numpy array to represent a small,
                rectangular portion of an image as a sliding 
                search window/kernel.
                
            startx: integer to represent window position start
                value on x-axis.
                
            starty: integer to represent window position start
                value on y-axis.
        
        """
        y, x = binary_window.nonzero()
        # get absolute coordinates in whole image
        y_absolute = y + starty
        x_absolute = x + startx
        n_samples = len(x) # number of samples
        x = x + 1
        if n_samples > 0:
            std = x.std()
        else:
            std = np.inf
        return n_samples, std, x_absolute.tolist(), y_absolute.tolist()


    ######################################################
    ### additional improvement for challenge videos    ###
    ######################################################
    def clean_inds(lane_ind_list, peak_list, threshold=50):
        """Clean noise from detected lane pixels."""

        #avg_peaks = np.sum(peak_list) // len(peak_list)
        x_inds = np.array(lane_ind_list[0])
        y_inds = np.array(lane_ind_list[1])
        #x = np.where(np.abs(x_inds-avg_peaks) > threshold)[0]
        #mask = np.ones(x_inds.shape[0], dtype=bool)
        #mask[x] = False
        #x_inds = x_inds[mask]
        #y_inds = y_inds[mask]

        return x_inds, y_inds
    
    
    warped, warped_histeq = warped_list

    if n_stepsx < 2:
        n_stepsx = 2
    offsety = warped.shape[0] // n_stepsy
    offsetx = warped.shape[1] // n_stepsx
    halfx = warped.shape[1] // 2
    
    margin = 50 # margin of detected cluster squares (just to draw a rectangle).
    
    # define output images for visualization
    out_img = np.zeros_like(warped)
    
    # list left and right lane indices
    left_lane_inds = [[],[]]
    right_lane_inds = [[],[]]

    # get binary images list
    binary_images = []
    
    # thresholds for combined color thresholding
    thr_hsv = (230,255)
    thr_luv = (157,255)
    thr_rgb_list = [
        (230, 255),
        (185, 230)
    ]
    for thr_rgb in thr_rgb_list:
        binary = combined_color_thresholding(
            warped, thr_rgb=thr_rgb, thr_hsv=thr_hsv, thr_luv=thr_luv)
        binary_images.append(binary)

    ######################################################
    ### additional improvement for severe shadow cases ###
    ######################################################
    bin_hsv_ch1, bin_hsv_ch2, bin_hsv_ch3 =\
        color_thresholding(warped_histeq, ch_type='hsv', thr=(0, 60), plot=False)
    # add binary image of H channel for severe shadow conditions
    binary_images.append(bin_hsv_ch1)
    thr_rgb_list = [
        (195, 255),
        (180, 255),
        (150, 200)
    ]
    for thr_rgb in thr_rgb_list:
        binary = combined_color_thresholding(
            warped_histeq, thr_rgb=thr_rgb, thr_hsv=(0,0), thr_luv=(0,0))
        binary_images.append(binary)
    
    img_size = (warped.shape[0], warped.shape[1])
    peaks_left = []
    peaks_right = []
    peak_left_prev = 522
    peak_right_prev = 800
    for i in range(n_stepsy):
        endy = img_size[0] - i*offsety
        starty = img_size[0] - (i+1)*offsety
        
        left_res, right_res = False, False
        for binary in binary_images:
            
            for j in range(n_stepsx):
                startx = j * offsetx
                endx = (j+1) * offsetx
                
                window = binary[starty:endy,startx:endx]
                
                n_samples, std, x, y = stdx(window, starty, startx)
                
                if startx < halfx:
                    found_flag = left_res
                else:
                    found_flag = right_res
                
                if (std < std_max) & (n_samples > min_samp) & (not found_flag):
                    histogram = np.sum(window, axis=0)
                    peak_base = np.argmax(histogram) + startx
                    winx_start = peak_base - margin
                    winx_end = peak_base + margin
                    # append x and y coors for detected lane pixels
                    # decide if left or right lane pixels
                    if peak_base < halfx:
                        color = (0,0,255)
                        if np.abs(peak_base - peak_left_prev) < 70:
                            peak_left_prev = peak_base
                            peaks_left.append(peak_base)
                            left_res = True
                            left_lane_inds[0].extend(x)
                            left_lane_inds[1].extend(y)
                            color = (255,0,0)
                    else:
                        color = (0,0,255)
                        if np.abs(peak_base - peak_right_prev) < 70:
                            peak_right_prev = peak_base
                            peaks_right.append(peak_base)
                            right_res = True
                            right_lane_inds[0].extend(x)
                            right_lane_inds[1].extend(y)
                            color = (0,255,0)
                    # draw output image
                    out_img[y,x,:] = 255
                    cv2.rectangle(
                        out_img,
                        (winx_start,starty),
                        (winx_end,endy),
                        color, 2)

            if left_res & right_res:
                # update midpoint of lanes (current_halfx)
                halfx = peaks_left[-1] + (peaks_right[-1]-peaks_left[-1])//2
                break

    # clean indices
    left_inds_x, left_inds_y = clean_inds(left_lane_inds, peaks_left, threshold=50)
    right_inds_x, right_inds_y = clean_inds(right_lane_inds, peaks_right, threshold=50)

    return out_img, left_inds_x, left_inds_y, right_inds_x, right_inds_y




# define a function for fitting and measurements
def fit_lane(lane_img, left_lane_inds, right_lane_inds):
    """Fit lane lines and do the measurements."""
    
    left_inds_x, left_inds_y = left_lane_inds
    right_inds_x, right_inds_y = right_lane_inds

    res = False
    marked_lane_img = None
    filled_lane_img = None
    fit_lane_img = None
    avg_curve_radi = None
    if (len(left_inds_x) > 0) & (len(left_inds_y) > 0)\
        & (len(right_inds_x) > 0) & (len(right_inds_y) > 0):
        
        res = True
        # fit
        # 2 degree polynomial
        # we will fit y, rather than x just because most x values may
        # be the same for different y values (f(y)=Ay^2+By+C).
        left_fit = np.polyfit(left_inds_y, left_inds_x, 2)
        right_fit = np.polyfit(right_inds_y, right_inds_x, 2)
    
        # generate x and y values for plotting
        ploty = np.linspace(0, lane_img.shape[0]-1, lane_img.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # mark and fill detected lanes
        # mark first
        zeros_img = np.zeros_like(lane_img[:,:,0]).astype(np.uint8)
        marked_lane_img = np.dstack((zeros_img, zeros_img, zeros_img))
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        mark_margin = 10
        left_line_window1 =\
            np.array([np.transpose(np.vstack([left_fitx-mark_margin, ploty]))])
        left_line_window2 =\
            np.array([np.flipud(np.transpose(np.vstack([left_fitx+mark_margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 =\
            np.array([np.transpose(np.vstack([right_fitx-mark_margin, ploty]))])
        right_line_window2 =\
            np.array([np.flipud(np.transpose(np.vstack([right_fitx+mark_margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(marked_lane_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(marked_lane_img, np.int_([right_line_pts]), (0,255, 0))
        marked_lane_img = cv2.addWeighted(lane_img, 1, marked_lane_img, 0.4, 0)
        
        # mark and fill
        filled_lane_img = np.dstack((zeros_img, zeros_img, zeros_img))
        # recast x and y to usable format
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(filled_lane_img, np.int_([pts]), (208, 255, 64))
    
        # measure the radious of the curvature
        # choose the max y value
        y_eval = np.max(ploty)
        # To apply pixel to meter conversion we will consider
        # lane is about 3.7 meters wide and 30 meters long.
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/200 # meters per pixel in x dimension
        left_fit_cr = np.polyfit(
            left_inds_y*ym_per_pix, left_inds_x*xm_per_pix, 2)
        right_fit_cr = np.polyfit(
            right_inds_y*ym_per_pix, right_inds_x*xm_per_pix, 2)
        left_curverad = (
            (1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5)\
            / np.absolute(2*left_fit_cr[0])
        right_curverad = (
            (1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5)\
            / np.absolute(2*right_fit_cr[0])
        avg_curve_radi = (left_curverad+right_curverad)*0.5 # in meters
    
    return res, marked_lane_img, filled_lane_img, avg_curve_radi





# A helper function to combine resulting images and add text
def combine_output(output_images, measurements):
    """Combine output images and write measurements."""
    
    combined_img = None
    if not None in output_images:
        # crop top views
        cropped_birds = bgr2rgb(output_images[0])[:,400:880]
        cropped_birds2 = bgr2rgb(output_images[2])[:,400:880]
        # concat 2 top view images and resize
        combined_birds = np.concatenate((cropped_birds2, cropped_birds), axis=1)
        combined_birds = cv2.resize(combined_birds, (combined_birds.shape[1]//2, combined_birds.shape[0]//2))
        # combine all 3 images
        combined_img = np.copy(bgr2rgb(output_images[4]))
        combined_img[0:combined_birds.shape[0],1280-combined_birds.shape[1]:1280] = combined_birds
        # add text
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        thickness = 2
        text = 'Predicted Lines'
        cv2.putText(combined_img, text, (807, 30), fontface, fontscale, (64,255,208), thickness)
        text = 'Top View'
        cv2.putText(combined_img, text, (1100, 30), fontface, fontscale, (64,255,208), thickness)
        text = 'Curve Radius = ' + '%.2f' % measurements[0] + 'm'
        cv2.putText(combined_img, text, (10, 30), fontface, fontscale, (64,255,208), thickness)
    
    return combined_img




def lane_detector(image, c_mtx=None, dist=None, std_max=15., min_samp=30):
    """Detect the lane and do the measurements.
    
    Steps:
    * Takes an image of road as the input.
    * Undistort the image with camera calibration.
    * Transforms the region of interest into bird's-eye view,
        gets a top-view of the road. Also returns this image.
    * Applies color thresholding to get a binary representation
        of lane lines.
    * Detects the lane line pixels and fits a second order polynomial 
        to those pixel coordinates. Returns the output image.
    * Measures the curvature radius of the lane and the position of
        the car in the lane and returns the results.
    * Colors the lane, transforms back to it's previous viewpoint
        and returns the final output image.
    
    Args:
        image: original road image in BGR colors, read with
            `cv2.imread`.
        
        c_mtx: Camera matrix for camera calibration.
        
        dist: Distortion coefficients for camera calibration.
        
        std_max: Float to represent maximum standard deviation of
            white pixel clusters to consider if they are lane pixels
            or not.
        
        min_samp: Integer to represent minimum number of white pixels
            required to consider if they are lane line pixels or not.
    """
    
    output_images = []
    measurements = []
    
    if (c_mtx is not None) & (dist is not None):
    	# undistort the raw image with camera calibration
    	image = undistort(image, c_mtx, dist)
    
    # source points for the top-view transformation
    points = np.array([
        [580, 457],
        [700, 457],
        [1280, 720],
        [0, 720],
        ])
    
    birds_eye_offsetx = 430
    # get top-view image (warped) and the inverse matrix (invmtx)
    invmtx, warped = birds_eye_transform(image, points, offsetx=birds_eye_offsetx)

    ######################################################
    ### additional improvement for severe shadow cases ###
    ######################################################
    ch1 = image[:,:,0] # blue channel
    ch2 = image[:,:,1] # green channel
    ch3 = image[:,:,2] # red channel
    # apply histogram equalization
    equ1 = cv2.equalizeHist(ch1)
    equ2 = cv2.equalizeHist(ch2)
    equ3 = cv2.equalizeHist(ch3)
    # combine channels back (in BGR order)
    histeq = np.dstack((equ1,equ2,equ3))
    # also get top-view of histeq
    _, warped_histeq = birds_eye_transform(histeq, points, offsetx=birds_eye_offsetx)

    # list histeq and warped
    warped_list = [bgr2rgb(warped), bgr2rgb(warped_histeq)]
    ######################################################
    
    # search lane pixels
    lane_lines, left_inds_x, left_inds_y, right_inds_x, right_inds_y =\
        search_lane(warped_list, n_stepsy=36, n_stepsx=2, std_max=std_max, min_samp=min_samp)
    
    # fit
    res, marked_lane_img, filled_lane_img, curve_radi =\
        fit_lane(lane_lines, (left_inds_x, left_inds_y), (right_inds_x, right_inds_y))
    
    # unwrap, get back to prev viewpoint and output the final image.
    img_size = image[:,:,0].shape[::-1]
    if res:
        unwarped = cv2.warpPerspective(filled_lane_img, invmtx, img_size)
        result_img = cv2.addWeighted(image, 1, unwarped, 0.5, 0)
    else:
        result_img = image
    
    output_images.append(warped)
    output_images.append(lane_lines)
    output_images.append(marked_lane_img)
    output_images.append(filled_lane_img)
    output_images.append(result_img)
    measurements.append(curve_radi)
    
    return output_images, measurements