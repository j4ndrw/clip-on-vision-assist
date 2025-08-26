import { useWifiStore } from "@/store";
import Chip from "@mui/material/Chip";
import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";
import WifiStrength from "./wifi-strength";
import Button from "@mui/material/Button";

const ScannedNetworks: React.FC<{ preconnectFn: () => void }> = ({ preconnectFn }) => {
  const { networks, selectNetwork } = useWifiStore();

  const handleConnectClick = (network: (typeof networks)[number]) => () => {
    preconnectFn();
    selectNetwork(network);
  };

  return (
    <Container
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "0.5rem",
      }}
    >
      {networks.length > 0 ? (
        <Typography variant="body1">Scanned networks</Typography>
      ) : null}
      {(networks ?? []).map((network, idx) => (
        <Container
          key={`${network.ssid}-${idx}`}
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "1rem",
          }}
        >
          <Chip
            variant="outlined"
            color="secondary"
            icon={<WifiStrength signal={network.signalStrengthDbm} />}
            label={network.ssid}
            sx={{ p: "0.5rem" }}
          />
          <Button variant="outlined" size="small" onClick={handleConnectClick(network)}>
            Connect
          </Button>
        </Container>
      ))}
    </Container>
  );
};

export default ScannedNetworks;
