import React, { useState, useCallback, useRef, useEffect } from "react";
import { Box, Text, useInput } from "ink";
import TextInput from "ink-text-input";
import { LlamaClient, Message } from "../api.js";
import { RetroSpinner } from "./RetroSpinner.js";
import { execSync } from "node:child_process";

interface ChatEntry {
  role: "user" | "assistant" | "system" | "info" | "error";
  content: string;
}

// Compare servers — check both ports for running models
// Model registry for compare mode
const MODEL_REGISTRY = [
  { id: "bonsai", name: "Bonsai-8B (1-bit)", port: 8203 },
  { id: "qwen06", name: "Qwen3-0.6B", port: 8210 },
  { id: "qwen17", name: "Qwen3-1.7B", port: 8211 },
  { id: "ministral", name: "Ministral-3B", port: 8212 },
  { id: "qwen4", name: "Qwen3-4B", port: 8213 },
  { id: "qwen9", name: "Qwen3.5-9B", port: 8204 },
  { id: "qwen8", name: "Qwen3-8B", port: 8214 },
];

interface ChatViewProps {
  client: LlamaClient;
  onQuit: () => void;
  onStatsUpdate: (stats: { tokPerSec: number; totalTokens: number }) => void;
  onStatusChange: (status: "connected" | "disconnected" | "streaming") => void;
}

const HELP_TEXT = `
╔══════════════════════════════════════════╗
║   tiny bit Commands                     ║
╠══════════════════════════════════════════╣
║ Just type to chat with the model        ║
║                                          ║
║ /search <query>  Web search (DuckDuckGo) ║
║ /image <path>    Describe an image       ║
║ /screenshot      Capture & describe      ║
║ /shell <cmd>     Run a shell command     ║
║ /stats           Show server stats       ║
║ /document <path> Parse PDF/doc/image       ║
║ /compare <prompt> Compare Bonsai vs 9B    ║
║ /clear           Clear chat history      ║
║ /models          List/download models     ║
║ /help            Show this help          ║
║ /quit            Exit tiny bit           ║
╚══════════════════════════════════════════╝`;

export const ChatView: React.FC<ChatViewProps> = ({
  client,
  onQuit,
  onStatsUpdate,
  onStatusChange,
}) => {
  const [entries, setEntries] = useState<ChatEntry[]>([
    {
      role: "system",
      content:
        '  Welcome to tiny bit. Type a message or /help for commands.',
    },
  ]);
  const [input, setInput] = useState("");
  const [showCommands, setShowCommands] = useState(false);
  const [spinnerMsg, setSpinnerMsg] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [scrollOffset, setScrollOffset] = useState(0);
  const messagesRef = useRef<Message[]>([
    {
      role: "system",
      content:
        `You are tiny bit, a helpful AI assistant running locally on Apple Silicon. Today is ${new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}. You respond concisely and helpfully.

You have access to these tools. To use a tool, include a tool_call tag in your response:

<tools>
[
  {"name": "shell", "description": "Run a shell command on macOS and see the output", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The shell command to execute"}}, "required": ["command"]}},
  {"name": "web_search", "description": "Search the web using DuckDuckGo", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}},
  {"name": "read_file", "description": "Read a text file or analyze an image/PDF. Supports .txt, .md, .py, .js, .json, .csv, .png, .jpg, .pdf and more.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path to read or analyze"}}, "required": ["path"]}}
]
</tools>

To call a tool, output EXACTLY this format (no other text before it):
<tool_call>{"name": "shell", "arguments": {"command": "ls ~/Desktop"}}</tool_call>

After the tool runs, you will receive the result and can then respond to the user.
Rules:
- When the user asks about files, folders, or their system, USE the shell tool directly. Do not just suggest commands.
- When the user asks about current events or news, USE the web_search tool.
- You can call multiple tools in sequence.
- After seeing tool results, give a clear summary to the user.`,
    },
  ]);
  const abortRef = useRef<AbortController | null>(null);

  // Visible entries for display (last N entries)
  const MAX_VISIBLE = 16;
  const visibleEntries = entries.slice(
    Math.max(0, entries.length - MAX_VISIBLE - scrollOffset),
    entries.length - scrollOffset > 0 ? entries.length - scrollOffset : undefined
  );

  useInput((ch, key) => {
    if (key.escape && isStreaming) {
      abortRef.current?.abort();
      setIsStreaming(false);
      setEntries((prev) => [
        ...prev,
        { role: "info", content: "◆ Generation stopped." },
      ]);
      onStatusChange("connected");
    }
    // Scroll with ctrl+up/down
    if (key.upArrow && key.ctrl) {
      setScrollOffset((prev) => Math.min(prev + 1, Math.max(0, entries.length - MAX_VISIBLE)));
    }
    if (key.downArrow && key.ctrl) {
      setScrollOffset((prev) => Math.max(prev - 1, 0));
    }
  });

  const handleSubmit = useCallback(
    async (value: string) => {
      const trimmed = value.trim();
      if (!trimmed) return;

      setInput("");
      setScrollOffset(0);

      // Handle commands
      if (trimmed.startsWith("/")) {
        const parts = trimmed.split(/\s+/);
        const cmd = parts[0].toLowerCase();
        const args = parts.slice(1).join(" ");

        switch (cmd) {
          case "/quit":
          case "/exit":
            onQuit();
            return;

          case "/help":
            setEntries((prev) => [
              ...prev,
              { role: "info", content: HELP_TEXT },
            ]);
            return;

          case "/clear":
            setEntries([
              {
                role: "system",
                content: "  Chat cleared.",
              },
            ]);
            messagesRef.current = [messagesRef.current[0]];
            return;

          case "/compare": {
            if (!args) {
              // Show available models and usage
              let online: string[] = [];
              let listing = "◆ /compare — Race two models!\n\n  Usage: /compare <model1> <model2> <prompt>\n\n  Available models:\n";
              for (const m of MODEL_REGISTRY) {
                let status = "○";
                try {
                  execSync(`curl -s --max-time 1 http://localhost:${m.port}/health 2>/dev/null | grep -q ok`, { timeout: 2000 });
                  status = "●";
                  online.push(m.id);
                } catch {}
                listing += `    ${status} ${m.id.padEnd(12)} ${m.name} (port ${m.port})\n`;
              }
              listing += `\n  Example: /compare bonsai qwen9 What is AI?\n`;
              if (online.length >= 2) {
                listing += `  Quick:   /compare ${online[0]} ${online[1]} What is AI?`;
              }
              setEntries((prev) => [...prev, { role: "info", content: listing }]);
              return;
            }

            // Parse: /compare <model1> <model2> <prompt>
            const parts = args.split(" ");
            let model1Id = "", model2Id = "", prompt = "";

            // Check if first two words match model IDs
            const m1 = MODEL_REGISTRY.find(m => m.id === parts[0]);
            const m2 = parts.length > 1 ? MODEL_REGISTRY.find(m => m.id === parts[1]) : null;

            if (m1 && m2) {
              model1Id = parts[0];
              model2Id = parts[1];
              prompt = parts.slice(2).join(" ");
            } else {
              // No model IDs — find any two online models
              const onlineModels: typeof MODEL_REGISTRY = [];
              for (const m of MODEL_REGISTRY) {
                try {
                  execSync(`curl -s --max-time 1 http://localhost:${m.port}/health 2>/dev/null | grep -q ok`, { timeout: 2000 });
                  onlineModels.push(m);
                } catch {}
              }
              if (onlineModels.length < 2) {
                setEntries((prev) => [...prev, { role: "error", content: "◆ Need 2+ models running. Use /models to see how to start them." }]);
                return;
              }
              model1Id = onlineModels[0].id;
              model2Id = onlineModels[1].id;
              prompt = args;
            }

            if (!prompt) {
              setEntries((prev) => [...prev, { role: "error", content: "◆ Need a prompt. Example: /compare bonsai qwen9 What is AI?" }]);
              return;
            }

            const server1 = MODEL_REGISTRY.find(m => m.id === model1Id)!;
            const server2 = MODEL_REGISTRY.find(m => m.id === model2Id)!;
            const servers = [
              { name: server1.name, url: `http://localhost:${server1.port}` },
              { name: server2.name, url: `http://localhost:${server2.port}` },
            ];

            setEntries((prev) => [
              ...prev,
              { role: "user", content: `/compare ${model1Id} vs ${model2Id}: ${prompt}` },
              { role: "info", content: `◆ Racing: ${server1.name} vs ${server2.name}` },
            ]);

            const compareMessages: Message[] = [
              { role: "system", content: "Answer concisely in 2-3 sentences." },
              { role: "user", content: prompt },
            ];

            const results: { [key: string]: string } = {};
            const cmpStats: { [key: string]: { tokPerSec: number; tokens: number } } = {};

            setIsStreaming(true);
            onStatusChange("streaming");

            const updateStreamDisplay = () => {
              const lines: string[] = [];
              for (const s of servers) {
                const text = results[s.name] || "";
                const st = cmpStats[s.name];
                lines.push(`┌─ ${s.name} ${st ? `✓ ${st.tokPerSec} tok/s` : "streaming..."}`);
                lines.push(`│ ${text || "..."}`);
                lines.push(`└${"─".repeat(50)}`);
              }
              setStreamingText(lines.join("\n"));
            };

            const promises = servers.map(async (server) => {
              try {
                const cmpClient = new LlamaClient(server.url);
                results[server.name] = "";
                await new Promise<void>((resolve) => {
                  cmpClient.streamChat(compareMessages, {
                    onToken: (token) => { results[server.name] += token; updateStreamDisplay(); },
                    onDone: (s) => { cmpStats[server.name] = { tokPerSec: s.tokensPerSec, tokens: s.totalTokens }; updateStreamDisplay(); resolve(); },
                    onError: (err) => { results[server.name] = `ERROR: ${err}`; updateStreamDisplay(); resolve(); },
                  });
                });
              } catch (e: any) {
                results[server.name] = `ERROR: ${e.message?.split("\n")[0]}`;
              }
            });

            await Promise.all(promises);

            // Show final results with winner
            const s1 = cmpStats[servers[0].name];
            const s2 = cmpStats[servers[1].name];
            const winner = s1 && s2 ? (s1.tokPerSec > s2.tokPerSec ? servers[0].name : servers[1].name) : "";

            for (const server of servers) {
              const s = cmpStats[server.name];
              const isWinner = server.name === winner;
              setEntries((prev) => [
                ...prev,
                { role: "info", content: `◆ ─── ${server.name} ${s ? `· ${s.tokPerSec} tok/s · ${s.tokens} tokens` : ""} ${isWinner ? "🏆" : ""} ───` },
                { role: "assistant", content: results[server.name] || "No response" },
              ]);
            }

            if (winner) {
              setEntries((prev) => [...prev, { role: "info", content: `◆ Winner: ${winner} (faster)` }]);
            }

            setIsStreaming(false);
            setStreamingText("");
            onStatusChange("connected");
            return;
          }

          case "/stats": {
            try {
              const health = await client.health();
              setEntries((prev) => [
                ...prev,
                {
                  role: "info",
                  content: `◆ Server Status: ${health.status}\n  Idle slots: ${health.slots_idle ?? "?"}\n  Processing: ${health.slots_processing ?? "?"}`,
                },
              ]);
            } catch (e: any) {
              setEntries((prev) => [
                ...prev,
                {
                  role: "error",
                  content: `◆ Error: ${e.message}`,
                },
              ]);
            }
            return;
          }

          case "/shell": {
            if (!args) {
              setEntries((prev) => [
                ...prev,
                { role: "error", content: "◆ Usage: /shell <task or command>" },
              ]);
              return;
            }
            setEntries((prev) => [
              ...prev,
              { role: "user", content: `/shell ${args}` },
            ]);

            // Check if it looks like a real command or natural language
            const looksLikeCommand = /^(ls|cd|cat|pwd|echo|grep|find|df|du|ps|top|whoami|date|uname|which|file|head|tail|wc|sort|mkdir|rm|cp|mv|chmod|chown|curl|wget|python|node|npm|git|brew|pip)\b/.test(args.trim());

            let cmd = args;
            if (!looksLikeCommand) {
              // Ask LLM to generate command
              setSpinnerMsg("Generating command...");
              try {
                const genResp = await client.chat([
                  { role: "system", content: "Generate a single macOS shell command for the user's request. Output ONLY the command, nothing else." },
                  { role: "user", content: args },
                ]);
                cmd = genResp.trim().replace(/^`+|`+$/g, "");
                setSpinnerMsg(`Executing: $ ${cmd}`);
                setEntries((prev) => [...prev, { role: "info", content: `◆ Running: $ ${cmd}` }]);
              } catch {
                setEntries((prev) => [...prev, { role: "error", content: "◆ Failed to generate command" }]);
                return;
              }
            }

            try {
              const output = execSync(cmd, {
                encoding: "utf-8",
                timeout: 10000,
                maxBuffer: 1024 * 1024,
              }).trim();
              setSpinnerMsg("");
              const truncOutput = output.split("\n").slice(0, 20).join("\n");
              setEntries((prev) => [
                ...prev,
                { role: "info", content: `◆ $ ${cmd}\n${truncOutput}` },
              ]);
              // Feed output to LLM for synthesis
              setSpinnerMsg("Summarizing...");
              messagesRef.current.push({
                role: "user",
                content: `I ran the shell command "$ ${cmd}" and got this output:\n\n${truncOutput}\n\nEach line above is a file or folder name. List them and briefly describe what you see. Do not say this is a screenshot — these are actual file/folder names returned by the command.`,
              });
              setIsStreaming(true);
              setStreamingText("");
              onStatusChange("streaming");
              abortRef.current = new AbortController();
              let fullText = "";
              await client.streamChat(messagesRef.current, {
                onToken: (token) => { setSpinnerMsg(""); fullText += token; setStreamingText(fullText); },
                onDone: (stats) => {
                  setSpinnerMsg("");
                  messagesRef.current.push({ role: "assistant", content: fullText });
                  setEntries((prev) => [...prev, { role: "assistant", content: fullText }]);
                  setIsStreaming(false);
                  setStreamingText("");
                  onStatsUpdate(stats);
                  onStatusChange("connected");
                },
                onError: (err) => {
                  setSpinnerMsg("");
                  setEntries((prev) => [...prev, { role: "error", content: `◆ Error: ${err}` }]);
                  setIsStreaming(false);
                  setStreamingText("");
                  onStatusChange("connected");
                },
              }, abortRef.current.signal);
            } catch (e: any) {
              setSpinnerMsg("");
              setEntries((prev) => [
                ...prev,
                {
                  role: "error",
                  content: `◆ Error: ${e.message?.split("\n")[0]}`,
                },
              ]);
            }
            return;
          }

          case "/search": {
            if (!args) {
              setEntries((prev) => [
                ...prev,
                { role: "error", content: "◆ Usage: /search <query>" },
              ]);
              return;
            }
            setEntries((prev) => [
              ...prev,
              { role: "user", content: `/search ${args}` },
            ]);
            setSpinnerMsg("Searching the web...");
            try {
              const safeQuery = args.replace(/[`$\\'"]/g, "");
              const isNews = /news|today|latest|recent|happened|breaking|update/i.test(args);
              const searchResults = execSync(
                `/opt/homebrew/bin/python3.13 -c "
import sys
from ddgs import DDGS
from datetime import datetime
query = sys.argv[1]
is_news = sys.argv[2] == '1'
results = []
with DDGS() as d:
    if is_news:
        for r in d.news(query, max_results=8):
            date = r.get('date', '')[:10]
            src = r.get('source', '')
            results.append('- [' + date + '] ' + r.get('title','') + ' (' + src + '): ' + r.get('body',''))
    else:
        dated_query = query + ' ' + datetime.now().strftime('%B %Y')
        for r in d.text(dated_query, max_results=8):
            results.append('- ' + r.get('title','') + ': ' + r.get('body',''))
print(chr(10).join(results))
" "${safeQuery}" "${isNews ? '1' : '0'}"`,
                { encoding: "utf-8", timeout: 15000 }
              ).trim();
              setSpinnerMsg("");
              if (searchResults) {
                setEntries((prev) => [
                  ...prev,
                  { role: "info", content: `◆ Found ${searchResults.split('\n').length} results:\n${searchResults.split('\n').map(l => '  ' + l.slice(0, 120)).join('\n')}` },
                ]);
                setSpinnerMsg("Synthesizing answer...");
                // Send results to LLM for synthesis
                const today = new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
                messagesRef.current.push({
                  role: "user",
                  content: `Today is ${today}. Summarize the key points from these search results. Be specific — include names, dates, and facts. Do not just list the sources.\n\nSearch results:\n${searchResults}\n\nUser's question: ${args}`,
                });
                // Stream LLM response (falls through to streaming below)
                setIsStreaming(true);
                setStreamingText("");
                onStatusChange("streaming");
                abortRef.current = new AbortController();
                let fullText = "";
                await client.streamChat(messagesRef.current, {
                  onToken: (token) => { setSpinnerMsg(""); fullText += token; setStreamingText(fullText); },
                  onDone: (stats) => {
                    setSpinnerMsg("");
                    messagesRef.current.push({ role: "assistant", content: fullText });
                    setEntries((prev) => [...prev, { role: "assistant", content: fullText }]);
                    setIsStreaming(false);
                    setStreamingText("");
                    onStatsUpdate(stats);
                    onStatusChange("connected");
                  },
                  onError: (err) => {
                    setEntries((prev) => [...prev, { role: "error", content: `◆ Error: ${err}` }]);
                    setIsStreaming(false);
                    setStreamingText("");
                    onStatusChange("connected");
                  },
                }, abortRef.current.signal);
                return;
              } else {
                setEntries((prev) => [...prev, { role: "info", content: "◆ No results found." }]);
              }
            } catch {
              setEntries((prev) => [
                ...prev,
                { role: "info", content: "◆ Search failed. Check: python3 search.py test" },
              ]);
            }
            return;
          }

          case "/image": {
            if (!args) {
              setEntries((prev) => [
                ...prev,
                { role: "error", content: "◆ Usage: /image <path>" },
              ]);
              return;
            }
            // Send as a chat message asking to describe
            setEntries((prev) => [
              ...prev,
              { role: "user", content: `[Describe image: ${args}]` },
            ]);
            messagesRef.current.push({
              role: "user",
              content: `Please describe what you think might be in an image at path: ${args}. Note: I cannot actually see images, but I can help discuss image-related topics.`,
            });
            break; // fall through to streaming
          }

          case "/document":
          case "/doc":
          case "/parse": {
            if (!args) {
              setEntries((prev) => [
                ...prev,
                { role: "error", content: "◆ Usage: /document <file path>\n  Supports: PDF, DOCX, XLSX, PPTX, PNG, JPG, TIFF" },
              ]);
              return;
            }
            const docPath = args.trim().replace(/^~/, process.env.HOME || "").replace(/['"]/g, "");
            setEntries((prev) => [...prev, { role: "user", content: `/document ${args}` }]);
            setSpinnerMsg(`LiteParse: ${docPath}`);

            let parsed = "";
            try {
              const docExt = docPath.split(".").pop() || "pdf";
              execSync(`cp "${docPath}" "/tmp/tinybit_doc.${docExt}"`, { timeout: 5000 });
              parsed = execSync(`npx lit parse "/tmp/tinybit_doc.${docExt}" --dpi 72 --num-workers 1 -q 2>/dev/null | head -300`, {
                encoding: "utf-8",
                timeout: 30000,
              }).trim();
            } catch (err: any) {
              setSpinnerMsg("");
              setEntries((prev) => [...prev, { role: "error", content: `◆ Parse failed: ${err.message?.split("\n")[0]}` }]);
              return;
            }

            setSpinnerMsg("");
            if (!parsed || parsed.length < 10) {
              setEntries((prev) => [...prev, { role: "info", content: "◆ No content extracted from document." }]);
              return;
            }

            setEntries((prev) => [...prev, {
              role: "info",
              content: `◆ Parsed ${parsed.split("\n").length} lines from ${args.split("/").pop()}`,
            }]);

            // Feed parsed content to the model
            messagesRef.current.push({
              role: "user",
              content: `I parsed this document (${args.split("/").pop()}) with LiteParse. Here is the extracted text:\n\n${parsed}\n\nSummarize the key contents of this document.`,
            });
            break; // fall through to agent loop
          }

          case "/models": {
            const modelList = [
              { name: "Qwen3-0.6B", size: "0.4 GB", speed: "50+ tok/s", repo: "unsloth/Qwen3-0.6B-GGUF", file: "Qwen3-0.6B-Q4_K_M.gguf", port: 8203 },
              { name: "Bonsai-8B (1-bit)", size: "1.16 GB", speed: "9-20 tok/s", repo: "prism-ml/Bonsai-8B-gguf", file: "Bonsai-8B.gguf", port: 8203, note: "needs PrismML llama.cpp" },
              { name: "Qwen3-1.7B", size: "1.1 GB", speed: "30+ tok/s", repo: "unsloth/Qwen3-1.7B-GGUF", file: "Qwen3-1.7B-Q4_K_M.gguf", port: 8203 },
              { name: "Ministral-3B", size: "2.15 GB", speed: "15-25 tok/s", repo: "lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF", file: "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", port: 8203 },
              { name: "Qwen3-4B", size: "2.5 GB", speed: "15-25 tok/s", repo: "unsloth/Qwen3-4B-GGUF", file: "Qwen3-4B-Q4_K_M.gguf", port: 8203 },
              { name: "Qwen3.5-9B (IQ2_XXS)", size: "3.19 GB", speed: "1-5 tok/s", repo: "unsloth/Qwen3.5-9B-GGUF", file: "Qwen3.5-9B-UD-IQ2_XXS.gguf", port: 8204 },
              { name: "Qwen3-8B", size: "5.0 GB", speed: "5-15 tok/s", repo: "unsloth/Qwen3-8B-GGUF", file: "Qwen3-8B-Q4_K_M.gguf", port: 8204 },
            ];
            let info = "◆ Available Models:\n\n";
            for (const m of modelList) {
              // Check if server is running on that port
              let status = "○ offline";
              try {
                execSync(`curl -s --max-time 1 http://localhost:${m.port}/health 2>/dev/null | grep -q ok`, { timeout: 2000 });
                status = "● online";
              } catch {}
              info += `  ${status} ${m.name}\n    Size: ${m.size} | Speed: ${m.speed}\n    Download: huggingface-cli download ${m.repo} ${m.file} --local-dir ./models\n\n`;
            }
            info += "  Switch model: restart with --server http://localhost:<port>";
            setEntries((prev) => [...prev, { role: "info", content: info }]);
            return;
          }

          case "/screenshot": {
            setEntries((prev) => [
              ...prev,
              { role: "user", content: "[Taking screenshot...]" },
            ]);
            try {
              execSync("screencapture -x /tmp/tinybit-screenshot.png", {
                timeout: 5000,
              });
              setEntries((prev) => [
                ...prev,
                {
                  role: "info",
                  content: "◆ Screenshot saved to /tmp/tinybit-screenshot.png",
                },
              ]);
            } catch {
              setEntries((prev) => [
                ...prev,
                {
                  role: "error",
                  content: "◆ Screenshot failed (requires macOS)",
                },
              ]);
            }
            return;
          }

          default:
            setEntries((prev) => [
              ...prev,
              {
                role: "error",
                content: `◆ Unknown command: ${cmd}. Type /help for commands.`,
              },
            ]);
            return;
        }
      }

      // Regular chat message
      if (!trimmed.startsWith("/")) {
        setEntries((prev) => [...prev, { role: "user", content: trimmed }]);
        messagesRef.current.push({ role: "user", content: trimmed });
      }

      // Agent loop: call LLM, check for tool calls, execute, feed back, repeat
      let loopCount = 0;
      const MAX_LOOPS = 5;

      while (loopCount < MAX_LOOPS) {
        loopCount++;
        setSpinnerMsg(loopCount === 1 ? "Thinking..." : "Processing tool results...");

        // Get non-streaming response to check for tool calls
        let response = "";
        try {
          response = await client.chat(messagesRef.current, 500);
        } catch (e: any) {
          setSpinnerMsg("");
          setEntries((prev) => [...prev, { role: "error", content: `◆ Error: ${e.message}` }]);
          break;
        }

        setSpinnerMsg("");

        // Check for tool_call in response
        const toolCallMatch = response.match(/<tool_call>([\s\S]*?)<\/tool_call>/);

        if (!toolCallMatch) {
          // No tool call — this is the final text response. Stream it for nice display.
          messagesRef.current.push({ role: "assistant", content: response });
          setEntries((prev) => [...prev, { role: "assistant", content: response }]);
          break;
        }

        // Parse and execute tool call
        try {
          const toolCall = JSON.parse(toolCallMatch[1]);
          const toolName = toolCall.name;
          const toolArgs = toolCall.arguments || {};

          // Show what the model is doing
          messagesRef.current.push({ role: "assistant", content: response });

          if (toolName === "shell") {
            const cmd = toolArgs.command;
            setSpinnerMsg(`Running: $ ${cmd}`);
            setEntries((prev) => [...prev, { role: "info", content: `◆ Running: $ ${cmd}` }]);

            let output = "";
            try {
              output = execSync(cmd, { encoding: "utf-8", timeout: 10000, maxBuffer: 1024 * 1024 }).trim();
            } catch (e: any) {
              output = `Error: ${e.message?.split("\n")[0]}`;
            }
            const truncOutput = output.split("\n").slice(0, 30).join("\n");
            setSpinnerMsg("");
            setEntries((prev) => [...prev, { role: "info", content: `◆ Output:\n${truncOutput}` }]);

            // Feed result back to model
            messagesRef.current.push({
              role: "user",
              content: `<tool_response>{"name": "shell", "output": ${JSON.stringify(truncOutput)}}</tool_response>`,
            });

          } else if (toolName === "web_search") {
            const query = toolArgs.query;
            setSpinnerMsg(`Searching: ${query}`);
            setEntries((prev) => [...prev, { role: "info", content: `◆ Searching: ${query}` }]);

            let results = "";
            try {
              const safeQuery = query.replace(/[`$\\'"]/g, "");
              const isNews = /news|today|latest|happened/i.test(query);
              results = execSync(
                `/opt/homebrew/bin/python3.13 -c "
import sys
from ddgs import DDGS
from datetime import datetime
q = sys.argv[1]
is_news = sys.argv[2] == '1'
res = []
with DDGS() as d:
    if is_news:
        for r in d.news(q, max_results=5):
            res.append(r.get('date','')[:10] + ' - ' + r.get('title','') + ': ' + r.get('body',''))
    else:
        for r in d.text(q + ' ' + datetime.now().strftime('%B %Y'), max_results=5):
            res.append(r.get('title','') + ': ' + r.get('body',''))
print(chr(10).join(res))
" "${safeQuery}" "${isNews ? '1' : '0'}"`,
                { encoding: "utf-8", timeout: 15000 }
              ).trim();
            } catch {
              results = "Search failed";
            }
            setSpinnerMsg("");
            setEntries((prev) => [...prev, { role: "info", content: `◆ Results:\n${results.split('\n').map(l => '  ' + l.slice(0, 100)).join('\n')}` }]);

            messagesRef.current.push({
              role: "user",
              content: `<tool_response>{"name": "web_search", "output": ${JSON.stringify(results)}}</tool_response>`,
            });

          } else if (toolName === "read_file") {
            let filePath = toolArgs.path.replace(/^~/, process.env.HOME || "");
            // Auto-resolve relative paths — check common locations
            if (!filePath.startsWith("/")) {
              const home = process.env.HOME || "";
              const candidates = [
                `${home}/Desktop/${filePath}`,
                `${home}/Downloads/${filePath}`,
                `${home}/Documents/${filePath}`,
                `${home}/${filePath}`,
                filePath,
              ];
              for (const c of candidates) {
                try { execSync(`test -f "${c}"`, { timeout: 1000 }); filePath = c; break; } catch {}
              }
            }
            const isDocument = /\.(pdf|docx|xlsx|pptx|doc|png|jpg|jpeg|tiff|bmp|heic)$/i.test(filePath);

            setSpinnerMsg(isDocument ? `LiteParse: ${filePath}` : `Reading: ${filePath}`);
            setEntries((prev) => [...prev, { role: "info", content: `◆ ${isDocument ? "Parsing" : "Reading"}: ${filePath}` }]);

            let content = "";
            try {
              if (isDocument) {
                // Use LiteParse — copy to temp to handle spaces in paths
                const ext = filePath.split(".").pop() || "pdf";
                execSync(`cp "${filePath}" "/tmp/tinybit_parse.${ext}"`, { timeout: 5000 });
                content = execSync(`npx lit parse "/tmp/tinybit_parse.${ext}" --dpi 72 --num-workers 1 -q 2>/dev/null | head -200`, {
                  encoding: "utf-8",
                  timeout: 30000,
                }).trim();
                if (!content || content.length < 10) {
                  content = "LiteParse returned no content. File may be empty or unsupported.";
                }
              } else {
                content = execSync(`head -80 "${filePath}"`, { encoding: "utf-8", timeout: 5000 }).trim();
              }
            } catch (err: any) {
              content = `Error: ${err.message?.split("\n")[0]}`;
            }
            setSpinnerMsg("");

            messagesRef.current.push({
              role: "user",
              content: `<tool_response>{"name": "read_file", "output": ${JSON.stringify(content)}}</tool_response>`,
            });

          } else {
            // Unknown tool
            setEntries((prev) => [...prev, { role: "error", content: `◆ Unknown tool: ${toolName}` }]);
            break;
          }
        } catch (e: any) {
          setEntries((prev) => [...prev, { role: "error", content: `◆ Tool parse error: ${e.message}` }]);
          // Still show the raw response
          messagesRef.current.push({ role: "assistant", content: response });
          setEntries((prev) => [...prev, { role: "assistant", content: response.replace(/<tool_call>[\s\S]*?<\/tool_call>/, '').trim() }]);
          break;
        }
      }

      onStatusChange("connected");
      setIsStreaming(false);
      setStreamingText("");
      setSpinnerMsg("");
    },
    [client, onQuit, onStatsUpdate, onStatusChange]
  );

  const getEntryColor = (role: ChatEntry["role"]) => {
    switch (role) {
      case "user":
        return "#FFBF00";
      case "assistant":
        return "#33FF33";
      case "system":
        return "#33FF33";
      case "info":
        return "#33FF33";
      case "error":
        return "red";
    }
  };

  const getEntryPrefix = (role: ChatEntry["role"]) => {
    switch (role) {
      case "user":
        return "▶ you";
      case "assistant":
        return " tiny bit";
      case "system":
        return "";
      case "info":
        return "";
      case "error":
        return "⚠";
    }
  };

  return (
    <Box flexDirection="column" flexGrow={1}>
      {/* Chat history */}
      <Box flexDirection="column" flexGrow={1} paddingX={1} width={Math.min((process.stdout.columns || 80) - 2, 78)}>
        {visibleEntries.map((entry, i) => (
          <Box key={i} flexDirection="column" marginBottom={1}>
            {entry.role !== "system" && entry.role !== "info" && entry.role !== "error" && (
              <Text color={getEntryColor(entry.role)} bold>
                {getEntryPrefix(entry.role)}:
              </Text>
            )}
            <Text
              color={getEntryColor(entry.role)}
              wrap="wrap"
              dimColor={entry.role === "info"}
            >
              {entry.role === "user" ? "   " : " "}{entry.content}
            </Text>
          </Box>
        ))}

        {/* Streaming response */}
        {spinnerMsg && !isStreaming && (
          <Box paddingLeft={2}>
            <RetroSpinner type="dots" label={spinnerMsg} />
          </Box>
        )}
        {isStreaming && (
          <Box flexDirection="column" marginBottom={1}>
            <Text color="#33FF33" bold>
              tiny bit:
            </Text>
            <Text color="#33FF33" wrap="wrap">
              {" "}{streamingText}
              <Text color="#33FF33">█</Text>
            </Text>
          </Box>
        )}
      </Box>

      {/* Input area */}
      <Box paddingX={1} marginTop={1}>
        <Text color="#1A8C1A">
          {"─".repeat(78)}
        </Text>
      </Box>
      <Box paddingX={1}>
        <Text color="#FFBF00" bold>
          ▶{" "}
        </Text>
        {isStreaming ? (
          <Text color="#33FF33" dimColor>
            (streaming... press ESC to stop)
          </Text>
        ) : (
          <TextInput
            value={input}
            onChange={(val) => {
              setInput(val);
              setShowCommands(val.startsWith("/") && !val.includes(" "));
            }}
            onSubmit={(val) => {
              setShowCommands(false);
              handleSubmit(val);
            }}
            placeholder="Type a message or /help..."
          />
        )}
      </Box>
      {showCommands && (() => {
        const cmds = [
          ["/search", "<query>", "web search + AI synthesis"],
          ["/shell", "<cmd>", "run a shell command"],
          ["/document", "<path>", "parse PDF/doc/image (LiteParse)"],
          ["/image", "<path>", "describe an image"],
          ["/screenshot", "", "capture + analyze screen"],
          ["/compare", "<prompt>", "compare Bonsai vs 9B"],
          ["/stats", "", "show performance stats"],
          ["/clear", "", "clear conversation"],
          ["/models", "", "list/download models"],
          ["/help", "", "show all commands"],
          ["/quit", "", "exit"],
        ].filter(([cmd]) => cmd.startsWith(input));
        return cmds.length > 0 ? (
          <Box flexDirection="column" paddingLeft={4}>
            <Text color="#1A8C1A">{"─".repeat(40)}</Text>
            {cmds.map(([cmd, arg, desc], i) => (
              <Text key={i} color="#33FF33"> {cmd} {arg ? arg + " " : ""}<Text color="#1A8C1A">{desc}</Text></Text>
            ))}
            <Text color="#1A8C1A">{"─".repeat(40)}</Text>
          </Box>
        ) : null;
      })()}
    </Box>
  );
};
