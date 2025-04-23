import pyrealsense2 as rs
import numpy as np
import cv2


def get_intrinsics():
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)

    # Get frames from the camera to get the intrinsic parameters
    profile = pipeline.get_active_profile()
    depth_stream = profile.get_stream(rs.stream.depth)
    intr = depth_stream.as_video_stream_profile().get_intrinsics()

    # Stop the pipeline
    pipeline.stop()

    # Intrinsics
    intrinsics_matrix = [     #  depth intrinsics
        [intr.fx, 0, intr.ppx],  #  [388.5013122558594, 0, 319.58245849609375]
        [0, intr.fy, intr.ppy],  #  [0, 388.5013122558594, 238.5020751953125]
        [0, 0, 1]                #  [0, 0, 1]
    ]

    return intrinsics_matrix

def show_rgb_depth():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays

            depth_image = np.asanyarray(depth_frame.get_data())

            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":

    # intrinsics_matrix = get_intrinsics()
    # print("Intrinsic matrix for RealSense D435 depth camera:")
    # for row in intrinsics_matrix:
    #     print(row)
    show_rgb_depth()

