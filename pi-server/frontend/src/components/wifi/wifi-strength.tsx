import {
  SignalWifi0Bar as SignalWifi0BarIcon,
  SignalWifi1Bar as SignalWifi1BarIcon,
  SignalWifi2Bar as SignalWifi2BarIcon,
  SignalWifi3Bar as SignalWifi3BarIcon,
  SignalWifi4Bar as SignalWifi4BarIcon,
} from "@mui/icons-material";

const categorizeWifiSignalStrength = (
  signal: number,
): "unreliable" | "poor" | "fair" | "good" | "excellent" => {
  if (signal < -85) return "unreliable";
  if (-85 <= signal && signal < -71) return "poor";
  if (-71 <= signal && signal < -60) return "fair";
  if (-60 <= signal && signal < -51) return "good";
  return "excellent";
};

const wifiSignalStrengthToIconMap: Record<
  ReturnType<typeof categorizeWifiSignalStrength>,
  () => React.JSX.Element
> = {
  unreliable: () => <SignalWifi0BarIcon />,
  poor: () => <SignalWifi1BarIcon />,
  fair: () => <SignalWifi2BarIcon />,
  good: () => <SignalWifi3BarIcon />,
  excellent: () => <SignalWifi4BarIcon />,
};

const WifiStrength: React.FC<{ signal: number }> = ({ signal }) => {
  const category = categorizeWifiSignalStrength(signal);
  const Icon = wifiSignalStrengthToIconMap[category];
  return <Icon />;
};
export default WifiStrength
