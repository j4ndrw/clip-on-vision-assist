import z from "zod";
import { responseSchema } from './response.schema';

export type GetBluetoothDevicesResponse = z.infer<typeof responseSchema>
