import cv2
import numpy as np
import sys


# function for rescale the image
def rescale_img(image):
    # define VGA size ie, rows = 480 (height) and columns= 600 (width)
    width = 600
    height = 480

    # Calculate the aspect ratio of the original image
    img_height, img_width, _ = image.shape
    img_aspect_ratio = img_width / img_height

    # check the required height and width matches with aspect ratio
    if img_aspect_ratio > (width / height):
        new_width = width
        new_height = int(width / img_aspect_ratio)
    else:
        new_height = height
        new_width = int(height * img_aspect_ratio)

    # Resize the image based on the aspect ratio
    rescaled_image = cv2.resize(image, (new_width, new_height))
    return rescaled_image

# function for generating sift keypoints a
def get_sift_image_with_circle(object, image):
    no_of_kps = object.detect(image, None)

    return no_of_kps


# function for generating sift keypoints and drawing circle over the image
def draw_the_line_and_circle(img, keypoints, org_img, file_name):
    for kp in keypoints:
        # get the x and y coordinates
        x_value = int(kp.pt[0])
        y_value = int(kp.pt[1])

        # drwaing lines
        # line_first = cv2.line(img, (x_value + 3, y_value), (x_value - 3, y_value), (), 1)
        line = cv2.line(img, (x_value, y_value + 3), (x_value, y_value - 3), (), 1)
    # draw and display the keypoints on the originl image
    resultant_image = cv2.drawKeypoints(line, sift_keypoints, input_img,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # showing output

    print("# of keypoints in {} is {}".format(file_name, len(keypoints)))
    output = np.hstack((org_img, resultant_image))
    cv2.imshow("Task-1-Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# implement K-means clustering
def calculate_kmeans_clustering(total_keypoints):
    k1 = int(0.05 * total_keypoints)
    k2 = int(0.10 * total_keypoints)
    k3 = int(0.20 * total_keypoints)

    return k1, k2, k3


# create dictionary to return values of percentage applied on each image
def get_sift_keypoionts_and_descriptors(key):
    k_perc_list = {0: 5, 1: 10, 2: 20}
    k_perc_out = k_perc_list[key]
    return k_perc_out

  #  implement histogram image
def create_histogram(hsgm_img):
    std_hist_op = np.linalg.norm(hsgm_img, ord=1)
    if (std_hist_op == 0):
        return hsgm_img

    return hsgm_img / std_hist_op

# Task 1 (if the user provides only one image, it will generate the SIFT keypoimts and draw the circle over it)
received_image = sys.argv
if (len(received_image)) == 2:
    # get the image and read it
    image = received_image[1]
    input_img = cv2.imread(image)
    ycrcb_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCrCb)

    # Step 1: Rescale the image
    # Rescale properly the image to a size comparable to VGA size (480(rows) x 600(columns)) to
    # reduce the computation. Note that the aspect ratio of the image should be kept when the image
    # is rescaled.

    # Resize the image based on the aspect ratio
    rescaled_image = rescale_img(ycrcb_image)
    rescaled_input_image = rescale_img(input_img)

    # to get the SIFT points convert the image to YCrCb image and split it to get Y value
    y_value, cr_value, cb_value = cv2.split(rescaled_image)

    # Step 2 and 3: Extract SIFT keypoints from the luminance Y component of the image
    # For each detected keypoint, draw a cross “+” at the location of the key point and a circle around
    # the keypoint whose radius is proportional to the scale of the keypoint and a line from the “+”
    # to the circle indicating the orientation of the key point.

    # create the sift
    sift_value = cv2.SIFT_create()

    # for detecting keypoints
    sift_keypoints = get_sift_image_with_circle(sift_value, y_value)

    # Step 4:Both the original image and the image with highlighted keypoints should be displayed in a
    # window

    # combine two images hoizontally
    sift_image_with_circle = draw_the_line_and_circle(y_value, sift_keypoints, rescaled_input_image, image)


else:
    # initialise variables
    keypoints_list = 0
    descriptors_list = []
    keypoints = []
    k_list = []
    # get all the images from the command prompt and add to a list
    imageList = sys.argv[1:]
    # Load and preprocess images, and extract SIFT descriptors
    for i in range(len(imageList)):
        # read each input image
        image = cv2.imread(imageList[i])
        # to get the SIFT points convert the image to YCrCb image and split it to get Y value
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Rescale the image
        rescle_img = rescale_img(ycrcb_img)
        # split img to get Y value
        y, cr, cb = cv2.split(rescle_img)
        # create the sift
        sift_in = cv2.SIFT_create()
        # to find the keypoints and descriptors by using direct method sift.detectAndCompute()
        kpoints, descr_list = sift_in.detectAndCompute(y, None)

        keypoints_list += len(kpoints)
        descriptors_list.append(descr_list)

        for kps in kpoints:
            keypoints.append(i)

        print("# of keypoints in {} is {}".format(imageList[i], len(kpoints)))
    # Perform K-means clustering
    cluster_val = np.vstack(descriptors_list)

    k_mean1, k_mean2, k_mean3 = calculate_kmeans_clustering(keypoints_list)

    k_list.append(k_mean1)
    k_list.append(k_mean2)
    k_list.append(k_mean3)
    kmeans = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    for inp in range(3):

        k_perc = get_sift_keypoionts_and_descriptors(inp)

        ret, placeholder, middle = cv2.kmeans(cluster_val, k_list[inp], None, kmeans, 10,
                                                    cv2.KMEANS_RANDOM_CENTERS)

        std_val = np.unique(placeholder)
        std_len_vl = len(std_val)

        # create histograms for all images
        histograms_list = []
        for j in range(0, len(imageList)):
            hstgrm = np.zeros(std_len_vl, dtype=np.float32)
            histograms_list.append(hstgrm)

        hisgm_img = np.vstack(histograms_list)

        for i in range(len(cluster_val)):
            lbl_1 = placeholder[i]
            lb_2 = keypoints[i]
            hisgm_img[lb_2][lbl_1] += 1
        # create histograms for all images
        for i in range(0, len(imageList)):
            hisgm_img[i] = create_histogram(hisgm_img[i])
        # get χ² distance
        # display the length of keypoints of each image,display Dissimilarity matrix and display cluster centers
        print("K = {} % * ({}) = {}".format(k_perc, keypoints_list, k_list[inp]))
        print("Dissimarilty matrix: ")
        for inpt in range(0, len(imageList)):
            for inp_i in range(0, len(imageList)):
                chi2_distances = cv2.compareHist(hisgm_img[inpt], hisgm_img[inp_i],
                                                     cv2.HISTCMP_CHISQR)
                print(round(chi2_distances, 2), end="    ")

            print("\n")

        print("\n")









