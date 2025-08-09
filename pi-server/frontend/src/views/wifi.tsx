import { useEffect, useRef, useState } from "react";

import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import { useWifiStore } from "@/store";
import { scanNetworks } from "@/services/wifi/scan-networks";
import ScannedNetworks from "@/components/wifi/scanned-networks";
import ConnectToNetwork from "@/components/wifi/connect-to-network-form";

function Wifi() {
  const scanNetworksInterval = useRef<NodeJS.Timeout>(null);

  const [scanning, setScanning] = useState(false);
  const { setNetworks, selectedNetwork } = useWifiStore();

  const handleScanNetworks = async () => {
    await scanNetworks({
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
      onSuccess: ({ wifiNetworks }) => setNetworks(wifiNetworks),
    });
  };

  const stopNetworkScan = () => {
    setScanning(false);
    if (scanNetworksInterval.current) {
      clearInterval(scanNetworksInterval.current);
    }
  };

  const startNetworkScan = async () => {
    stopNetworkScan();

    setScanning(true);

    await handleScanNetworks();
    scanNetworksInterval.current = setInterval(handleScanNetworks, 10000);
  };

  useEffect(() => {
    return () => {
      stopNetworkScan();
    };
  }, []);

  useEffect(() => {
    if (selectedNetwork) stopNetworkScan();
  }, [selectedNetwork]);

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
      <Typography variant="h6">Wi-Fi Configuration</Typography>
      {!scanning && (
        <Button onClick={startNetworkScan} variant="outlined" color="primary">
          Start Wi-Fi Scan
        </Button>
      )}
      {scanning && (
        <Button onClick={stopNetworkScan} variant="outlined" color="secondary">
          Stop Wi-Fi Scan
        </Button>
      )}

      <ScannedNetworks preconnectFn={stopNetworkScan} />
      {scanning && (
        <Container
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            gap: "1rem",
          }}
        >
          <Typography variant="overline">Scanning for networks</Typography>
          <CircularProgress color="info" size="1rem" />
        </Container>
      )}
      <ConnectToNetwork />
    </Container>
  );
}

export default Wifi;
