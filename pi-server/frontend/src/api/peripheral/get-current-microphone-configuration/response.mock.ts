import type { GetCurrentMicrophoneConfigurationResponse } from "./types";

export const mock: GetCurrentMicrophoneConfigurationResponse = {
  microphoneConfig: {
    audioCaptureConfig: { secondsPerChunk: 1, maxChunks: 25 },
    silenceDetectionConfig: {
      minSilenceLenMs: 2000,
      silenceThresholdDbfs: -25,
    },
  },
};
