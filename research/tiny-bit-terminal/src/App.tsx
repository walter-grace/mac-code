import React, { useState, useCallback, useEffect } from "react";
import { Box, Text } from "ink";
import { BootScreen } from "./components/BootScreen.js";
import { ChatView } from "./components/ChatView.js";
import { StatusBar } from "./components/StatusBar.js";
import { LlamaClient } from "./api.js";

interface AppProps {
  serverUrl: string;
}

type AppPhase = "boot" | "chat";

const HEADER = `╔══════════════════════════════════════════════════════════════════════════════╗
║   tiny bit ── Local AI Agent                                       v1.0  ║
╠══════════════════════════════════════════════════════════════════════════════╣`;

export const App: React.FC<AppProps> = ({ serverUrl }) => {
  const [phase, setPhase] = useState<AppPhase>("boot");
  const [modelName, setModelName] = useState("loading...");
  const [tokPerSec, setTokPerSec] = useState(0);
  const [totalTokens, setTotalTokens] = useState(0);
  const [serverStatus, setServerStatus] = useState<
    "connected" | "disconnected" | "streaming"
  >("disconnected");
  const [shouldExit, setShouldExit] = useState(false);

  const client = React.useMemo(
    () => new LlamaClient(serverUrl),
    [serverUrl]
  );

  // Fetch model name on connect
  useEffect(() => {
    const fetchModel = async () => {
      try {
        const name = await client.getModelName();
        setModelName(name);
        setServerStatus("connected");
      } catch {
        setModelName("(offline)");
        setServerStatus("disconnected");
      }
    };
    if (phase === "chat") {
      fetchModel();
    }
  }, [client, phase]);

  const handleBootComplete = useCallback(() => {
    setPhase("chat");
  }, []);

  const handleQuit = useCallback(() => {
    setShouldExit(true);
    setTimeout(() => process.exit(0), 100);
  }, []);

  const handleStatsUpdate = useCallback(
    (stats: { tokPerSec: number; totalTokens: number }) => {
      setTokPerSec(stats.tokPerSec);
      setTotalTokens((prev) => prev + stats.totalTokens);
    },
    []
  );

  if (shouldExit) {
    return (
      <Box flexDirection="column" alignItems="center" padding={2}>
        <Text color="#33FF33" bold>
          ╔══════════════════════════════════╗
        </Text>
        <Text color="#33FF33" bold>
          ║                                  ║
        </Text>
        <Text color="#33FF33" bold>
          ║   It is now safe to turn off    ║
        </Text>
        <Text color="#33FF33" bold>
          ║       your computer.            ║
        </Text>
        <Text color="#33FF33" bold>
          ║                                  ║
        </Text>
        <Text color="#33FF33" bold>
          ╚══════════════════════════════════╝
        </Text>
      </Box>
    );
  }

  if (phase === "boot") {
    return (
      <Box flexDirection="column">
        <BootScreen onComplete={handleBootComplete} serverUrl={serverUrl} />
      </Box>
    );
  }

  return (
    <Box flexDirection="column" width={Math.min(process.stdout.columns || 80, 80)}>
      {/* Header */}
      <Box flexDirection="column">
        <Text color="#33FF33" wrap="truncate">{HEADER}</Text>
      </Box>

      {/* Chat area */}
      <ChatView
        client={client}
        onQuit={handleQuit}
        onStatsUpdate={handleStatsUpdate}
        onStatusChange={setServerStatus}
      />

      {/* Status bar */}
      <Box paddingX={1}>
        <StatusBar
          modelName={modelName}
          tokPerSec={tokPerSec}
          totalTokens={totalTokens}
          serverStatus={serverStatus}
          serverUrl={serverUrl}
        />
      </Box>
    </Box>
  );
};
