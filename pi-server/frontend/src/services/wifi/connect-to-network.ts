import { wifiApi } from "@/api";
import { createMutationService } from "../utils";

export const connectToNetwork = createMutationService(wifiApi.connectToNetwork);
