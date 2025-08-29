import { DEV_API_HOST } from "@/api/constants";

export const CAMERA_FEED_SOURCE = import.meta.env.DEV
  ? `${DEV_API_HOST}/api/testbed/camera/feed`
  : "/api/testbed/camera/feed";
