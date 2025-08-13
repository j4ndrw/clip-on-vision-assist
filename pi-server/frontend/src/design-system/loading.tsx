import CircularProgress from "@mui/material/CircularProgress";
import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";

type Props = {
  title: string
}

const Loading = ({ title }: Props) => (
  <Container
    sx={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      gap: "1rem",
    }}
  >
    <Typography variant="overline">{title}</Typography>
    <CircularProgress color="info" size="1rem" />
  </Container>
);

export default Loading
