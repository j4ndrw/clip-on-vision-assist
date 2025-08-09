import { useEffect, useRef, useState } from "react";

import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import { useBluetoothStore } from "@/store";
import { scanBluetoothDevices } from "@/services/bluetooth/scan-bluetooth-devices";
import ScannedBluetoothDevices from "@/components/bluetooth/scanned-bluetooth-devices";

function Bluetooth() {
  const scanBluetoothDevicesInterval = useRef<NodeJS.Timeout>(null);

  const [scanning, setScanning] = useState(false);
  const { setDevices } = useBluetoothStore();

  const handleScanBluetoothDevices = async () => {
    await scanBluetoothDevices({
      onValidationError: (error) => {
        // TODO
        console.log(error);
        return;
      },
      onApiError: (error) => {
        // TODO
        console.log(error);
        return;
      },
      onSuccess: ({ bluetoothDevices }) => setDevices(bluetoothDevices),
    });
  };

  const stopBluetoothScan = () => {
    setScanning(false);
    if (scanBluetoothDevicesInterval.current) {
      clearInterval(scanBluetoothDevicesInterval.current);
    }
  };

  const startBluetoothScan = async () => {
    stopBluetoothScan();

    setScanning(true);

    await handleScanBluetoothDevices();
    scanBluetoothDevicesInterval.current = setInterval(handleScanBluetoothDevices, 10000);
  };

  useEffect(() => {
    return () => {
      stopBluetoothScan();
    };
  }, []);

  return (
    <Container
      sx={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        gap: "2rem",
      }}
    >
      <Typography variant="h6">Bluetooth Settings</Typography>
      {!scanning && (
        <Button onClick={startBluetoothScan} variant="outlined" color="primary">
          Start Bluetooth Scan
        </Button>
      )}
      {scanning && (
        <Button onClick={stopBluetoothScan} variant="outlined" color="secondary">
          Stop Bluetooth Scan
        </Button>
      )}

      <ScannedBluetoothDevices preconnectFn={stopBluetoothScan} />
      {scanning && (
        <Container
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            gap: "1rem",
          }}
        >
          <Typography variant="overline">Scanning for bluetooth devices</Typography>
          <CircularProgress color="info" size="1rem" />
        </Container>
      )}
    </Container>
  );
}

export default Bluetooth;
