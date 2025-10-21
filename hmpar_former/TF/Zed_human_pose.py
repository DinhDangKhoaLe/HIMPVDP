#!/usr/bin/env python3
# zed_human_pose.py
import sys
import cv2
import numpy as np
import pyzed.sl as sl


def pick_available_enum(enum_cls, candidates):
    """Return the first enum member that exists in enum_cls from candidates list (by name)."""
    for name in candidates:
        if hasattr(enum_cls, name):
            return getattr(enum_cls, name)
    return None


class ZEDHumanPose:
    """
    Initialize a ZED2/ZED2i, detect human joints, and draw them on the left image.
    Works with both:
      - Newer SDKs (Body Tracking API: BodyTrackingParameters/BODY_TRACKING_MODEL, retrieve_bodies)
      - Older SDKs (Object Detection API: ObjectDetectionParameters/OBJECT_DETECTION_MODEL, retrieve_objects)
    """

    def __init__(self,
                 resolution=None,
                 depth_mode=None,
                 show_window=True,
                 fps_limit=30):
        self.show_window = show_window
        self.fps_limit = max(1, int(fps_limit))

        # --- Camera open ---
        self.cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution or getattr(sl.RESOLUTION, "HD720")
        init_params.depth_mode = depth_mode or getattr(sl.DEPTH_MODE, "PERFORMANCE")
        init_params.coordinate_units = getattr(sl.UNIT, "MILLIMETER")
        init_params.coordinate_system = getattr(sl.COORDINATE_SYSTEM, "IMAGE")

        status = self.cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {repr(status)}")

        # --- Positional tracking (required for tracking IDs/temporal consistency) ---
        track_params = sl.PositionalTrackingParameters()
        status = self.cam.enable_positional_tracking(track_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Enable positional tracking failed: {repr(status)}")

        # Detect which API we have
        self.has_body_tracking_api = hasattr(sl, "BodyTrackingParameters") and hasattr(sl, "BODY_TRACKING_MODEL")

        # Common runtime
        self.runtime_params = sl.RuntimeParameters()

        # Prepare image buffer
        self.left = sl.Mat()

        # Body format (18 vs 34)
        self.body_format = pick_available_enum(getattr(sl, "BODY_FORMAT"), ["BODY_34", "BODY_18"])

        # Skeleton connectivity (simple/robust)
        self.edges_18 = [
            (0,1),(1,2),(2,3),(3,4),
            (1,5),(5,6),(6,7),
            (1,8),(8,9),(9,10),
            (1,11),(11,12),(12,13),
            (1,14),(14,15),(1,16),(16,17)
        ]
        self.edges_34 = [
            (0,1),(1,2),(2,3),(3,4),
            (1,5),(5,6),(6,7),            # right arm
            (1,8),(8,9),(9,10),           # left arm
            (1,11),(11,12),(12,13),       # right leg
            (1,14),(14,15),(15,16),       # left leg
            (2,17),(5,18),(8,19),(11,20)  # torso extras (indices may vary by SDK; safe to skip if out of range)
        ]

        if self.has_body_tracking_api:
            self._setup_body_tracking_api()
        else:
            self._setup_object_detection_api()

    # ---------- Setup paths ----------

    def _setup_body_tracking_api(self):
        """Newer SDK path: Body Tracking API."""
        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = True
        if self.body_format:
            body_params.body_format = self.body_format
        # pick a detection model that exists
        body_params.detection_model = pick_available_enum(
            sl.BODY_TRACKING_MODEL,
            ["HUMAN_BODY_FAST", "HUMAN_BODY_MEDIUM", "HUMAN_BODY_ACCURATE"]
        ) or sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE

        # Optional smoothing/fitting if available
        if hasattr(body_params, "enable_body_fitting"):
            body_params.enable_body_fitting = True

        status = self.cam.enable_body_tracking(body_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Enable body tracking failed: {repr(status)}")

        # Runtime + container
        if hasattr(sl, "BodyTrackingRuntimeParameters"):
            self.bt_rt = sl.BodyTrackingRuntimeParameters()
            if hasattr(self.bt_rt, "detection_confidence_threshold"):
                self.bt_rt.detection_confidence_threshold = 40
        else:
            self.bt_rt = None

        self.bodies = sl.Bodies()

        # Accessors for keypoints
        self.mode = "body_tracking"

    def _setup_object_detection_api(self):
        """Older SDK path: Object Detection API with HUMAN_BODY_* models."""
        od_params = sl.ObjectDetectionParameters()
        od_params.enable_tracking = True
        od_params.enable_segmentation = False

        # Prefer human body model if available, otherwise raise upgrade hint
        model = pick_available_enum(
            sl.OBJECT_DETECTION_MODEL,
            ["HUMAN_BODY_FAST", "HUMAN_BODY_MEDIUM", "HUMAN_BODY_ACCURATE"]
        )
        if model is None:
            raise RuntimeError(
                "Your ZED SDK doesn't expose HUMAN_BODY_* under OBJECT_DETECTION_MODEL.\n"
                "Upgrade to ZED SDK 4.x/5.x (or a 3.7+ release with body models)."
            )
        od_params.detection_model = model

        # Some SDKs expose body_format here as well
        if hasattr(od_params, "body_format") and self.body_format:
            od_params.body_format = self.body_format
        if hasattr(od_params, "enable_body_fitting"):
            od_params.enable_body_fitting = True

        status = self.cam.enable_object_detection(od_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Enable object detection (body) failed: {repr(status)}")

        self.od_rt = sl.ObjectDetectionRuntimeParameters()
        if hasattr(self.od_rt, "detection_confidence_threshold"):
            self.od_rt.detection_confidence_threshold = 40

        self.objects = sl.Objects()

        self.mode = "object_detection"

    # ---------- Helpers ----------

    @staticmethod
    def _valid_xy(p):
        return (p[0] > 0) and (p[1] > 0)

    def _pick_edges(self, n_kp):
        if n_kp >= 34:
            return self.edges_34
        if n_kp >= 18:
            return self.edges_18
        return [(i, i + 1) for i in range(max(0, n_kp - 1))]

    def _draw_skeleton(self, img, kps2d):
        if kps2d is None or len(kps2d) == 0:
            return
        pts = np.asarray(kps2d, dtype=np.int32).reshape(-1, 2)
        edges = self._pick_edges(len(pts))

        # joints
        for x, y in pts:
            if self._valid_xy((x, y)):
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        # limbs
        for a, b in edges:
            if a < len(pts) and b < len(pts):
                pa, pb = pts[a], pts[b]
                if self._valid_xy(pa) and self._valid_xy(pb):
                    cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                             (255, 0, 0), 2, lineType=cv2.LINE_AA)

    # ---------- Main loop ----------

    def run(self):
        delay_ms = max(1, int(1000 / self.fps_limit))

        while True:
            if self.cam.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
                # intermittent failures happen; don't crash the loop
                cv2.waitKey(1)
                continue

            self.cam.retrieve_image(self.left, sl.VIEW.LEFT)
            frame = self.left.get_data()  # BGRA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if self.mode == "body_tracking":
                if self.bt_rt is not None and hasattr(self.cam, "retrieve_bodies"):
                    self.cam.retrieve_bodies(self.bodies, self.bt_rt)
                else:
                    self.cam.retrieve_bodies(self.bodies)

                if self.bodies.is_new:
                    for body in self.bodies.body_list:
                        # try the most common attributes across SDKs
                        kp2d = None
                        if hasattr(body, "keypoint_2d") and len(body.keypoint_2d) > 0:
                            kp2d = body.keypoint_2d
                        elif hasattr(body, "skeleton_2d") and len(body.skeleton_2d) > 0:
                            kp2d = body.skeleton_2d
                        elif hasattr(body, "skeleton") and hasattr(body.skeleton, "keypoints_2d"):
                            kp2d = body.skeleton.keypoints_2d
                        self._draw_skeleton(frame, kp2d)

                        # optional: 2D bbox if provided
                        if hasattr(body, "bounding_box_2d") and len(body.bounding_box_2d) == 4:
                            bb = np.int32(body.bounding_box_2d)
                            for i in range(4):
                                p1 = tuple(bb[i % 4])
                                p2 = tuple(bb[(i + 1) % 4])
                                cv2.line(frame, p1, p2, (0, 255, 255), 1)

            else:  # object_detection fallback
                self.cam.retrieve_objects(self.objects, self.od_rt)
                if self.objects.is_new:
                    for obj in self.objects.object_list:
                        # Filter to PERSON if label exists
                        if hasattr(sl, "OBJECT_CLASS"):
                            is_person = (obj.label == sl.OBJECT_CLASS.PERSON)
                        else:
                            is_person = True  # older SDKs: assume body model already filters

                        if is_person:
                            kp2d = None
                            if hasattr(obj, "keypoint_2d") and len(obj.keypoint_2d) > 0:
                                kp2d = obj.keypoint_2d
                            elif hasattr(obj, "skeleton_2d") and len(obj.skeleton_2d) > 0:
                                kp2d = obj.skeleton_2d
                            elif hasattr(obj, "skeleton") and hasattr(obj.skeleton, "keypoints_2d"):
                                kp2d = obj.skeleton.keypoints_2d
                            self._draw_skeleton(frame, kp2d)

                            if hasattr(obj, "bounding_box_2d") and len(obj.bounding_box_2d) == 4:
                                bb = np.int32(obj.bounding_box_2d)
                                for i in range(4):
                                    p1 = tuple(bb[i % 4])
                                    p2 = tuple(bb[(i + 1) % 4])
                                    cv2.line(frame, p1, p2, (0, 255, 255), 1)

            if self.show_window:
                cv2.imshow("ZED2 Human Pose", frame)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord('q')):
                    break
            else:
                # headless mode: you could add a callback to deliver frames outward
                pass

        self.close()

    # ---------- Cleanup ----------

    def close(self):
        try:
            if self.has_body_tracking_api and hasattr(self.cam, "disable_body_tracking"):
                self.cam.disable_body_tracking()
            elif hasattr(self.cam, "disable_object_detection"):
                self.cam.disable_object_detection()
        except Exception:
            pass
        try:
            self.cam.disable_positional_tracking()
        except Exception:
            pass
        self.cam.close()
        if self.show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app = ZEDHumanPose(show_window=True, fps_limit=30)
        app.run()
    except Exception as e:
        print("Error:", e)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        sys.exit(1)
