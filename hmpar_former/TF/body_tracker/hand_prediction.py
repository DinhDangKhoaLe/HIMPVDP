import cv2
import time
import pyzed.sl as sl
import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
WINDOW_SIZE = 15
PREDICT_FRAMES = 15

# Base to camera transform (given)
transform_matrix_base_to_camera = np.array([
    [-0.9995, -0.0171, -0.0268, 0.8206],
    [0.0266, 0.0098, -0.9996, 0.4035],
    [0.0173, -0.9998, -0.0094, 0.7805],
    [0, 0, 0, 1]
])

# Compute camera to base transform
transform_matrix_camera_to_base = np.linalg.inv(transform_matrix_base_to_camera)

def transform_points_to_base(points, transform_matrix):
    """
    Transform points from camera frame to base frame
    points: np.array of shape (N,3)
    transform_matrix: 4x4 homogeneous transform
    """
    if points is None or len(points) == 0:
        return None
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])  # (N,4)
    points_base_h = (transform_matrix @ points_h.T).T
    return points_base_h[:, :3]

class ZEDRightHandPredictor3D:
    def __init__(self):
        print("Initializing ZED2 Camera for Right Hand Prediction (3D Visualization)...")
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Failed to open ZED camera.")
            exit(1)

        # Enable positional and body tracking
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        positional_tracking_parameters.set_as_static = True
        self.zed.enable_positional_tracking(positional_tracking_parameters)

        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = True
        body_params.enable_body_fitting = True
        body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        body_params.body_format = sl.BODY_FORMAT.BODY_18
        self.zed.enable_body_tracking(body_params)

        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        self.body_runtime_param.detection_confidence_threshold = 80

        self.bodies = sl.Bodies()
        self.image = sl.Mat()
        self.right_hand_positions = deque(maxlen=WINDOW_SIZE)

        # Setup matplotlib 3D plot
        plt.ion()
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("Right Hand Trajectory (Base Frame)")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        # Static axis limits relative to base frame
        self.static_xlim = (-0.5, 0.5)
        self.static_ylim = (-0.5, 0.5)
        self.static_zlim = (-1.0, -0.2)
        self.ax.set_xlim(*self.static_xlim)
        self.ax.set_ylim(*self.static_ylim)
        self.ax.set_zlim(*self.static_zlim)

    def predict_future_positions(self, positions):
        positions = np.array(positions)
        if positions.shape[0] < WINDOW_SIZE:
            return None
        X = np.arange(len(positions)).reshape(-1, 1)
        future_steps = np.arange(len(positions), len(positions) + PREDICT_FRAMES).reshape(-1, 1)

        preds = []
        for dim in range(3):
            model = LinearRegression()
            model.fit(X, positions[:, dim])
            y_future = model.predict(future_steps)
            preds.append(y_future)

        return np.stack(preds, axis=1)
        
    def update_3d_plot(self, current_points, predicted_points=None):
        self.ax.cla()
        self.ax.set_title("Right Hand Trajectory (ZED2)")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        # Reapply static axis limits
        self.ax.set_xlim(*self.static_xlim)
        self.ax.set_ylim(*self.static_ylim)
        self.ax.set_zlim(*self.static_zlim)

        current_points = np.array(current_points)
        if len(current_points) > 0:
            self.ax.plot(current_points[:, 0], current_points[:, 1], current_points[:, 2],
                         'ro-', label="Current Right Hand")

        if predicted_points is not None:
            self.ax.plot(predicted_points[:, 0], predicted_points[:, 1], predicted_points[:, 2],
                         'bo--', label="Predicted Future")

        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

    def run(self):
        print("Starting tracking... Press 'q' to quit.")
        print("Waiting for body tracking to start...")

        for _ in range(30):
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_bodies(self.bodies, self.body_runtime_param)
                if len(self.bodies.body_list) > 0:
                    break
            time.sleep(0.1)

        while True:
            if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            err = self.zed.retrieve_bodies(self.bodies, self.body_runtime_param)
            if err != sl.ERROR_CODE.SUCCESS:
                continue

            if len(self.bodies.body_list) > 0:
                for body in self.bodies.body_list:
                    if body.confidence < 80:
                        continue

                    right_hand = body.keypoint[4]
                    if not np.isnan(right_hand[0]):
                        self.right_hand_positions.append(right_hand)

                        predicted_positions = None
                        if len(self.right_hand_positions) == WINDOW_SIZE:
                            predicted_positions = self.predict_future_positions(self.right_hand_positions)

                        # # Transform points to base frame
                        # current_points_base = transform_points_to_base(
                        #     np.array(self.right_hand_positions), transform_matrix_camera_to_base
                        # )
                        # predicted_points_base = None
                        # if predicted_positions is not None:
                        #     predicted_points_base = transform_points_to_base(predicted_positions, transform_matrix_camera_to_base)

                        # Plot in base frame
                        # self.update_3d_plot(current_points_base, predicted_points_base)
                        
                        # Also plot in camera frame
                        self.update_3d_plot(self.right_hand_positions, predicted_positions)

            if cv2.waitKey(1) == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        print("Cleaning up...")
        self.image.free(sl.MEM.CPU)
        self.zed.disable_body_tracking()
        self.zed.disable_positional_tracking()
        self.zed.close()
        plt.ioff()
        plt.close(self.fig)
        print("Camera closed and resources released.")

if __name__ == "__main__":
    tracker = ZEDRightHandPredictor3D()
    tracker.run()
