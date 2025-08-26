import { wifiApi } from "@/api";
import { createQueryService } from "../utils";

export const scanNetworks = createQueryService(wifiApi.scanNetworks);
