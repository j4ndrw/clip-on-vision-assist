import { useState } from "react";
import { useMemo } from "react";

import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";
import Chip from "@mui/material/Chip";

type Props<
  TTabs extends Array<{
    id: string;
    Icon: () => React.JSX.Element;
    label: string;
  }>,
> = {
  title: string;
  tabs: TTabs;
  views: Record<TTabs[number]["id"], () => React.JSX.Element>;
};

function TabBasedRouter<
  TTabs extends Array<{
    id: string;
    Icon: () => React.JSX.Element;
    label: string;
  }>,
>({ title, tabs, views }: Props<TTabs>) {
  const [activeTab, setActiveTab] =
    useState<(typeof tabs)[number]["id"]>(tabs[0].id);

  const handleTabClick = (idx: number) => () => {
    setActiveTab(tabs[idx].id);
  };

  const CurrentView: () => React.JSX.Element = useMemo(() => views[activeTab], [views, activeTab]);

  return (
    <Container
      maxWidth="sm"
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "2rem",
        paddingBlock: "2rem",
        height: "100vh",
      }}
    >
      <Container
        sx={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          gap: "2rem",
        }}
      >
        <Typography variant="h6">
          {title}
        </Typography>
        <Container
          sx={{
            display: "flex",
            flexWrap: 'wrap',
            gap: "0.5rem 0.5rem",
            justifyContent: "center"
          }}
        >
          {tabs.map((tab, idx) => (
            <Chip
              sx={{ padding: "1rem" }}
              key={tab.id}
              color={tab.id === activeTab ? "primary" : "default"}
              variant={tab.id === activeTab ? "filled" : "outlined"}
              onClick={handleTabClick(idx)}
              label={tab.label}
              icon={<tab.Icon />}
            />
          ))}
        </Container>
      </Container>
      <CurrentView />
    </Container>
  );
}

export default TabBasedRouter;
