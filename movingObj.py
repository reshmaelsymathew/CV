import math
import cv2
import numpy as np
import sys


def main():

    #  get estimated  background frame
    def create_background_estimation(input_video):
        # get the frame count
        FOI_value = input_video.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

        # initialize empty array
        frames = []
        for frameOI in FOI_value:
            input_video.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
            _, frame = input_video.read()
            frames.append(frame)

        backgroundg_frame = np.median(frames, axis=0).astype(dtype=np.uint8)

        return backgroundg_frame

    # downscling
    def downscale_video(image, max_width, max_height):
        frame_width = image.shape[1]
        frame_height = image.shape[0]

        fx = max_width / frame_width
        fy = max_height / frame_height

        if (fx < fy):
            scalingFactor = fx
        else:
            scalingFactor = fy

        frame = cv2.resize(image, (0, 0), fx=scalingFactor, fy=scalingFactor, interpolation=cv2.INTER_LINEAR)
        return frame

    if sys.argv[1] == '-b':
        # get the video argument
        user_input_video = sys.argv[2]

        # Open the video file
        input_video = cv2.VideoCapture(user_input_video)

        if input_video.isOpened() == False:
            print("Error while opening the file!")

        # Get the estimated background frame
        estimated_background_frame = create_background_estimation(input_video)

        #Creating GMM model
        GMM_bg_subtractor_ip = cv2.createBackgroundSubtractorMOG2(
            varThreshold=25, detectShadows=False)
        input_video_captured = cv2.VideoCapture(cv2.samples.findFile(user_input_video))

        frame_count = 0
        while True:
            _, frame = input_video_captured.read()

            if frame is None:
                break
            # creating  mask
            foreground_mask_GMM = GMM_bg_subtractor_ip.apply(frame)

            # removing noise from mask
            kernel = np.ones((3, 3), np.uint8)
            image_mask_value = foreground_mask_GMM > 0
            black_color_background = np.zeros_like(frame, np.uint8)
            black_color_background[image_mask_value] = frame[image_mask_value]
            opening_value = cv2.morphologyEx(black_color_background, cv2.MORPH_OPEN, kernel)
            closing_value = cv2.morphologyEx(opening_value, cv2.MORPH_CLOSE, kernel)

            # stacking
            row1 = np.hstack([frame, estimated_background_frame])
            row2 = np.hstack(
                [np.repeat(foreground_mask_GMM[:, :, np.newaxis], 3, axis=2), closing_value])
            combined_output = np.vstack([row1, row2])
            cv2.namedWindow('GMM', cv2.WINDOW_NORMAL)
            cv2.imshow('GMM', combined_output)

            # Detection for connected components
            kernel_value = np.ones((100, 100), np.uint8)
            closing_gray_value = cv2.cvtColor(closing_value, cv2.COLOR_BGR2GRAY)
            dialated_frame_ip = cv2.dilate(closing_gray_value, kernel_value, iterations=1)
            eroded_frame_ip = cv2.erode(dialated_frame_ip, kernel_value, iterations=1)
            output_frame = cv2.medianBlur(eroded_frame_ip, 7)
            _, threshold_value = cv2.threshold(
                output_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # connectivity
            connectivity_value = 8
            result = cv2.connectedComponentsWithStats(
                threshold_value, connectivity_value, cv2.CV_32S)
            labels_count = result[0]
            stats = result[2]

            # printing categorised values on console
            frame_count = frame_count + 1
            car = 0
            person = 0
            other = 0
            for x in range(labels_count):
                if x != 0:
                    horizontal = stats[x][2]
                    vertical = stats[x][3]
                    if (horizontal > vertical) and horizontal > 40 and vertical > 30:
                        car = car + 1
                    elif (vertical > horizontal) and vertical > 20:
                        person = person + 1
                    else:
                        other = other + 1
            if (labels_count - 1) > 0:
                print('Frame ', frame_count, ' : ', labels_count - 1, 'objects',
                      '( ', person, ' persons, ', car, ' car, ', 'and ', other, ' others )')
            else:
                print('Frame ', frame_count, ' : ', '0 objects')
            # press escape to close window
            if cv2.waitKey(30) == 27:
                break

        input_video.release()
        cv2.destroyAllWindows()



    elif sys.argv[1] == '-d':
        # get the video argument
        user_input_video = sys.argv[2]

        # load the DNN model
        dnn = cv2.dnn.readNet(model='frozen_inference_graph.pb', config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                              framework='TensorFlow')
        input_video = cv2.VideoCapture(user_input_video)

        while True:
            _, frame_value1 = input_video.read()
            _, frame_value2 = input_video.read()
            _, frame_value3 = input_video.read()
            _, frame_value4 = input_video.read()

            if (frame_value1 is None) or (frame_value2 is None) or (frame_value3 is None) or (frame_value4 is None):
                break

            # downscale video
            frame_value1 = downscale_video(frame_value1, 600, 400)
            frame_value2 = downscale_video(frame_value2, 600, 400)
            frame_value3 = downscale_video(frame_value3, 600, 400)
            frame_value4 = downscale_video(frame_value4, 600, 400)

            if _:
                vframe_1 = frame_value2
                vframe_2 = frame_value3
                vframe_3 = frame_value4

                # load the input image for the model
                blob = cv2.dnn.blobFromImage(image=vframe_1, size=(300, 300), mean=(104, 117, 123), swapRB=True)

                dnn.setInput(blob)
                op_value = dnn.forward()

                # Perform object detection
                iteration_value = 1
                closed_Objects = {}
                for detection in op_value[0, 0, :, :]:
                    conf_ip = detection[2]
                    class_id = detection[1]
                    # Adjust the confidence threshold
                    if conf_ip > .35 and class_id == 1:
                        color = (0, 255, 225, 3)
                        # Calculate x and y of the corner of the bounding box
                        x = int(detection[3] * vframe_1.shape[1])
                        y = int(detection[4] * vframe_1.shape[0])

                        # Calculate the height and width of the box
                        _width = int(detection[5] * vframe_1.shape[1])
                        _height = int(detection[6] * vframe_1.shape[0])

                        cv2.rectangle(vframe_1, (x, y), (_width, _height), color, thickness=1)
                        cv2.rectangle(vframe_2, (x, y), (_width, _height), color, thickness=1)
                        distance = int(math.dist((x, y), (600, 400)))
                        closed_Objects[distance] = (x, y, _width, _height)
                        cv2.putText(vframe_2, f'{iteration_value}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 2, 225), 2)
                        iteration_value += 1

                sorted_objects = sorted(closed_Objects.items())
                count = 1
                for obj in sorted_objects:
                    if count <= 3:
                        cv2.rectangle(vframe_3, (int(obj[1][0]), int(obj[1][1])), (int(obj[1][2]), int(obj[1][3])),
                                      (0, 255, 1, 3), 2)
                        cv2.putText(vframe_3, 'distance = {}'.format(int(obj[0])), (int(obj[1][0]), int(obj[1][1]) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 1), 2)
                    else:
                        break
                    count += 1
                output = np.vstack([np.hstack([frame_value1, vframe_1]), np.hstack([vframe_2, vframe_3])])
                cv2.imshow("Result", output)
                
                # Enter to quit
                if cv2.waitKey(75) == 13:
                    break
            else:
                break

        input_video.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()