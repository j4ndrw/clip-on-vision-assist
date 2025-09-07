from src.control_center.models.base import BaseSchema
from src.env import Environment, environment


class CameraConfig(BaseSchema):
    num_frames_to_capture: int
    fps: int
    wait_for_next_batch_factor: int

    @staticmethod
    def from_environment(environment: Environment = environment):
        env = environment.get()

        peripheral_camera_num_frames_to_capture = env.get(
            "PERIPHERAL_CAMERA_NUM_FRAMES_TO_CAPTURE", None
        )
        peripheral_camera_fps = env.get("PERIPHERAL_CAMERA_FPS", None)
        peripheral_camera_wait_for_next_batch_factor = env.get(
            "PERIPHERAL_CAMERA_WAIT_FOR_NEXT_BATCH_FACTOR", None
        )

        assert peripheral_camera_num_frames_to_capture is not None, (
            "`PERIPHERAL_CAMERA_NUM_FRAMES_TO_CAPTURE` environment variable is not defined!"
        )
        assert peripheral_camera_fps is not None, (
            "`PERIPHERAL_CAMERA_FPS` environment variable is not defined!"
        )
        assert peripheral_camera_wait_for_next_batch_factor is not None, (
            "`PERIPHERAL_CAMERA_WAIT_FOR_NEXT_BATCH_FACTOR` environment variable is not defined!"
        )

        return CameraConfig(
            num_frames_to_capture=int(peripheral_camera_num_frames_to_capture),
            fps=int(peripheral_camera_fps),
            wait_for_next_batch_factor=int(
                peripheral_camera_wait_for_next_batch_factor
            ),
        )

    def save_to_environment(self, environment: Environment = environment):
        environment.update(
            key="PERIPHERAL_CAMERA_NUM_FRAMES_TO_CAPTURE",
            value=str(self.num_frames_to_capture),
        ).update(key="PERIPHERAL_CAMERA_FPS", value=str(self.fps)).update(
            key="PERIPHERAL_CAMERA_WAIT_FOR_NEXT_BATCH_FACTOR",
            value=str(self.wait_for_next_batch_factor),
        )
