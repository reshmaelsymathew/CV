import cv2
import sys
import numpy as np


def main():
    if len(sys.argv) == 3 and sys.argv[1] in ['-XYZ', '-Lab', '-HSB', '-YCrCb']:
        # Assignment 1 -Task 1
        # Add a condition to check whether the input command is to convert the image to diffrent color space or not
        print("Convert image {1} to color space {0} ".format(sys.argv[1], sys.argv[2]))

        required_color_space = sys.argv[1]
        input_image = sys.argv[2]

        # check whether the image is available or not
        if input_image is None:
            print("Error: Could not read the input image.")
            return

        # initialise the width and height of image to display
        width = 400
        height = 300

        # read the input image
        image = cv2.imread(input_image)
        resized_image = cv2.resize(image, (width, height))

        # Convert the image to grayscale, display and print its pixel values
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(grayscale_image)
        resized_gray_scale_image = cv2.resize(grayscale_image, (width, height))
        cv2.imshow("Gray Scale image", resized_gray_scale_image)

        # Convert to the specified color space
        if required_color_space == "-XYZ":
            # convert original image to XYZ color space
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)

            # Split the XYZ components
            x, y, z = cv2.split(converted_image)

            # Create a canvas to display first row as original image , x component
            first_row = np.zeros((image.shape[0], image.shape[1] * 2, 3), dtype=np.uint8)
            first_row[:, :image.shape[1]] = image
            first_row[:, image.shape[1]:] = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

            # Create a canvas to display second row as y component ,z  component
            second_row = np.zeros((image.shape[0], image.shape[1] * 2, 3), dtype=np.uint8)
            second_row[:, :image.shape[1]] = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
            second_row[:, image.shape[1]:] = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)

            # Combine the two rows into a single window
            combined_canvas = np.vstack((first_row, second_row))

            # display the merged image
            cv2.imshow('Original image with XYZ color components', combined_canvas)

        elif required_color_space == "-HSB":
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Split the HSV components
            h, s, v = cv2.split(converted_image)

            # Create a canvas to display first row as original image , h component
            first_row = np.hstack((image, cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)))

            # Create a canvas to display second row as s component , v component
            second_row = np.hstack((cv2.cvtColor(s, cv2.COLOR_GRAY2BGR), cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)))

            # Combine the two rows into a single window
            combined_canvas = np.vstack((first_row, second_row))

            # display the merged image
            cv2.imshow('Original Image and HSV Components', combined_canvas)

        elif required_color_space == "-Lab":
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

            # Split the Lab components
            l, a, b = cv2.split(converted_image)

            # Create a canvas to display first row as original image , L component
            first_row = np.hstack((image, cv2.cvtColor(l, cv2.COLOR_GRAY2BGR)))

            # Create a canvas to display second row as a component , b component
            second_row = np.hstack((cv2.cvtColor(a, cv2.COLOR_GRAY2BGR), cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)))

            # Combine the two rows into a single window
            combined_canvas = np.vstack((first_row, second_row))

            # display the merged image
            cv2.imshow('Original Image and Lab Components', combined_canvas)

        elif required_color_space == "-YCrCb":
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

            # Split the YCrCb components
            y, cr, cb = cv2.split(converted_image)

            # Create a canvas to display first row as original image , Y component
            first_row = np.hstack((image, cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)))

            # Create a canvas to display second row as Cr component , Cb component
            second_row = np.hstack((cv2.cvtColor(cr, cv2.COLOR_GRAY2BGR), cv2.cvtColor(cb, cv2.COLOR_GRAY2BGR)))

            # Combine the two rows into a single window
            combined_canvas = np.vstack((first_row, second_row))

            # display the merged image
            cv2.imshow('Original Image and YCrCb Components', combined_canvas)
        else:
            print("Error: Invalid color space specified.")
            return
        # bind the image and finally destroy all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Assignment 1 - Task 2
        # Check whether the command is to display the image on the scenic image
    else:
        print("Display the image {0} over the scenic image {1}".format(sys.argv[2], sys.argv[1]))

        # assign both person image and scenic image to variables
        scenic_image_received = sys.argv[1]
        person_image_received = sys.argv[2]

        # Read both person with green screen image and scenic image
        green_screen_image = cv2.imread(person_image_received)
        scenic_image = cv2.imread(scenic_image_received)
        scene_image = cv2.imread(scenic_image_received)

        # Convert green screen image to HSV color space
        converted_hsv_image = cv2.cvtColor(green_screen_image, cv2.COLOR_BGR2HSV)

        # get the lower and upper bounds for the green color in HSV
        lower_green_value = np.array([35, 100, 100])
        upper_green_value = np.array([85, 255, 255])

        # Create a mask for the green pixels
        green_mask = cv2.inRange(converted_hsv_image, lower_green_value, upper_green_value)

        # Invert the mask to get non-green pixels
        non_green_mask = cv2.bitwise_not(green_mask)

        # Extract the person's image using the non-green mask
        person_extracted_image = cv2.bitwise_and(green_screen_image, green_screen_image, mask=non_green_mask)

        # Convert the person image to a transparent image
        converted_transparent_image = cv2.cvtColor(person_extracted_image, cv2.COLOR_BGR2BGRA)

        # Create an alpha channel for the transparent person image
        alpha_channel = np.zeros_like(non_green_mask)
        alpha_channel[non_green_mask > 0] = 255

        # Set the alpha channel for transparency
        converted_transparent_image[:, :, 3] = alpha_channel

        # Determine the position to overlay the transparent image
        x_position = 100  # Adjust as needed
        y_position = 100  # Adjust as needed

        # Calculate the dimensions of the overlay region
        overlay_height, overlay_width = converted_transparent_image.shape[:2]

        # Code for display image on white background
        # Create a white background image with the same height as the person in the green background image
        white_background = np.ones_like(person_extracted_image) * 255

        # Ensure the overlay position is within the white background image
        if x_position + overlay_width > white_background.shape[1]:
            overlay_width = white_background.shape[1] - x_position
        if y_position + overlay_height > white_background.shape[0]:
            overlay_height = white_background.shape[0] - y_position

        # Extract the relevant region from the scenic image
        white_region = white_background[y_position:y_position + overlay_height,
                       x_position:x_position + overlay_width]

        # Overlay the transparent image onto the white background
        for y in range(overlay_height):
            for x in range(overlay_width):
                alpha = converted_transparent_image[y, x, 3] / 255.0
                white_region[y, x] = alpha * converted_transparent_image[y, x, :3] + (1 - alpha) * white_region[y, x]

        # code to display the image on white background ends here

        # code for image ovelay on scenic image started
        # Convert the person image to a transparent image
        transparent_person_image = cv2.cvtColor(person_extracted_image, cv2.COLOR_BGR2BGRA)

        # Create an alpha channel for the transparent person image
        alpha_channel = np.zeros_like(non_green_mask)
        alpha_channel[non_green_mask > 0] = 255

        # Set the alpha channel for transparency
        transparent_person_image[:, :, 3] = alpha_channel

        # Determine the position to overlay the transparent image
        x_position = 100  # Adjust as needed
        y_position = 100  # Adjust as needed

        # Calculate the dimensions of the overlay region
        overlay_height, overlay_width = transparent_person_image.shape[:2]

        # Ensure the overlay position is within the scenic image boundaries
        if x_position + overlay_width > scenic_image.shape[1]:
            overlay_width = scenic_image.shape[1] - x_position
        if y_position + overlay_height > scenic_image.shape[0]:
            overlay_height = scenic_image.shape[0] - y_position

        # Extract the relevant region from the scenic image
        scenic_region = scenic_image[y_position:y_position + overlay_height,
                        x_position:x_position + overlay_width]

        # Overlay the transparent image onto the scenic image
        for y in range(overlay_height):
            for x in range(overlay_width):
                alpha = transparent_person_image[y, x, 3] / 255.0
                scenic_region[y, x] = alpha * transparent_person_image[y, x, :3] + (1 - alpha) * scenic_region[y, x]

        # resize all the converted color space images to same dimension
        width = 400
        height = 300
        original_person_image = cv2.resize(green_screen_image, (width, height))
        original_scenic_image = cv2.resize(scene_image, (width, height))
        person_with_white_background = cv2.resize(white_region, (width, height))
        person_with_scenic_background = cv2.resize(scenic_region, (width, height))

        # combine two images hoizontally
        first_row_with_green_image = cv2.hconcat([original_person_image, person_with_white_background])
        second_row_with_scenic_image = cv2.hconcat([original_scenic_image, person_with_scenic_background])
        # vertically stack the combined row images
        merged_image_of_green_and_scenic_image = cv2.vconcat([first_row_with_green_image, second_row_with_scenic_image])

        cv2.imshow("Combined with scenic pic", merged_image_of_green_and_scenic_image)
        # bind the image and finally destroy all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()