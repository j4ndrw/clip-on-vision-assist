import type { GetCurrentCameraConfigurationResponse } from "./types";

export const mock: GetCurrentCameraConfigurationResponse = {
  cameraConfig: {
    numFramesToCapture: 2,
    fps: 1,
    waitForNextBatchFactor: 2
  }
}
