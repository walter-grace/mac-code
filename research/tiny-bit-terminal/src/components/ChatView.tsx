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

// ── PicoClaw-inspired: Tool Registry with TTL + Discovery ──
interface ToolDef {
  name: string;
  description: string;
  parameters: any;
  hidden?: boolean;  // hidden tools are discoverable but not in every prompt
  ttl?: number;      // remaining turns before auto-demotion back to hidden
}

// Core tools (always visible) — sorted alphabetically for deterministic KV cache
const CORE_TOOLS: ToolDef[] = [
  { name: "read_file", description: "Read a text file or parse a document (PDF/DOCX/image). Supports .txt, .md, .py, .js, .json, .csv, .png, .jpg, .pdf, .docx and more.", parameters: { type: "object", properties: { path: { type: "string", description: "File path to read or analyze" } }, required: ["path"] } },
  { name: "search_tools", description: "Find additional tools by keyword. Use when you need a capability not listed above.", parameters: { type: "object", properties: { query: { type: "string", description: "What you need to do, e.g. 'write file' or 'list directory'" } }, required: ["query"] } },
  { name: "shell", description: "Run a shell command on macOS and see the output", parameters: { type: "object", properties: { command: { type: "string", description: "The shell command to execute" } }, required: ["command"] } },
  { name: "web_search", description: "Search the web for current information. Use specific technical terms, not natural language questions.", parameters: { type: "object", properties: { query: { type: "string", description: "Specific search query with precise keywords" } }, required: ["query"] } },
];

// Hidden tools (discoverable via search_tools) — promoted with TTL when found
const HIDDEN_TOOLS: ToolDef[] = [
  { name: "write_file", hidden: true, description: "Write content to a file, creating it if needed", parameters: { type: "object", properties: { path: { type: "string", description: "File path to write" }, content: { type: "string", description: "Content to write" } }, required: ["path", "content"] } },
  { name: "list_dir", hidden: true, description: "List files and folders in a directory with details", parameters: { type: "object", properties: { path: { type: "string", description: "Directory path (default: home)" } }, required: [] } },
  { name: "screenshot", hidden: true, description: "Capture a screenshot of the current screen", parameters: { type: "object", properties: {} } },
  { name: "system_info", hidden: true, description: "Get system information: CPU, RAM, disk, OS version", parameters: { type: "object", properties: {} } },
];

const TOOL_PROMOTE_TTL = 5; // turns before auto-demotion

// ── PicoClaw-inspired: Context Budget Management ──
const CONTEXT_BUDGET = 1800; // tokens — leave headroom below llama-server's -c 2048
const CHARS_PER_TOKEN = 2.5; // PicoClaw heuristic

function estimateTokens(messages: Message[]): number {
  let chars = 0;
  for (const m of messages) {
    if (typeof m.content === "string") chars += m.content.length;
    else chars += JSON.stringify(m.content).length;
  }
  return Math.ceil(chars / CHARS_PER_TOKEN);
}

function trimToContextBudget(messages: Message[], budget: number): Message[] {
  if (estimateTokens(messages) <= budget) return messages;

  // Always keep system prompt (index 0)
  const system = messages[0];
  let rest = messages.slice(1);

  // Drop oldest messages in groups of 2 (user + assistant pairs)
  // Never split tool-call sequences: user -> assistant(tool_call) -> user(tool_response)
  while (estimateTokens([system, ...rest]) > budget && rest.length > 2) {
    // Find safe cut point — don't split mid-tool-sequence
    let cutIdx = 0;
    for (let i = 0; i < rest.length - 1; i++) {
      const content = typeof rest[i].content === "string" ? rest[i].content : "";
      // Safe to cut before a user message that isn't a tool_response
      if (rest[i].role === "user" && !content.includes("<tool_response>")) {
        cutIdx = i;
        break;
      }
      // Also safe to cut before a plain user message
      if (i > 0 && rest[i].role === "user") {
        cutIdx = i;
        break;
      }
    }
    // Remove from cutIdx, at least 2 messages
    const removeCount = Math.max(2, cutIdx + 1);
    rest = rest.slice(removeCount);
  }

  return [system, ...rest];
}

function buildToolsPrompt(coreTools: ToolDef[], promotedHidden: ToolDef[]): string {
  const allVisible = [...coreTools, ...promotedHidden].sort((a, b) => a.name.localeCompare(b.name));
  return JSON.stringify(allVisible.map(t => ({ name: t.name, description: t.description, parameters: t.parameters })), null, 2);
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
  { id: "gemma4", name: "Gemma 4-26B MoE", port: 8205 },
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
║ /compare         Race two models          ║
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

  // ── PicoClaw-inspired: Promoted hidden tools with TTL ──
  const promotedToolsRef = useRef<ToolDef[]>([]);

  // ── PicoClaw-inspired: Steering Queue — queue user input during tool execution ──
  const steeringQueueRef = useRef<string[]>([]);
  const isProcessingRef = useRef(false);

  // Build system prompt with current tool set
  const buildSystemPrompt = useCallback(() => {
    const toolsJson = buildToolsPrompt(CORE_TOOLS, promotedToolsRef.current);
    return `You are tiny bit, a helpful AI assistant running locally on Apple Silicon. Today is ${new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}. You respond concisely and helpfully.

You have access to these tools. To use a tool, include a tool_call tag in your response:

<tools>
${toolsJson}
</tools>

To call a tool, output EXACTLY this format (no other text before it):
<tool_call>{"name": "shell", "arguments": {"command": "ls ~/Desktop"}}</tool_call>

You can call MULTIPLE tools at once by including multiple tool_call tags:
<tool_call>{"name": "shell", "arguments": {"command": "ls ~/Desktop"}}</tool_call>
<tool_call>{"name": "shell", "arguments": {"command": "date"}}</tool_call>

After the tool runs, you will receive the result and can then respond to the user.
Rules:
- When the user asks about files, folders, or their system, USE the shell tool directly. Do not just suggest commands.
- When the user asks about current events or news, USE the web_search tool. Use SPECIFIC keywords, not vague phrases. For example: instead of "today's news in the LLM world", search "LLM large language model news March 2026" or "OpenAI Anthropic Google AI announcements". Do multiple targeted searches if the topic is broad.
- You can call multiple tools in sequence or in parallel.
- After seeing tool results, give a clear summary to the user.
- If you need a tool not listed above, use search_tools to find it.`;
  }, []);

  const messagesRef = useRef<Message[]>([
    { role: "system", content: "" }, // placeholder — updated on first use
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
                // Pre-check: is server even reachable?
                const cmpClient = new LlamaClient(server.url);
                try {
                  await cmpClient.health();
                } catch {
                  results[server.name] = `OFFLINE — server not running on ${server.url}`;
                  updateStreamDisplay();
                  return;
                }
                results[server.name] = "";
                await new Promise<void>((resolve, reject) => {
                  cmpClient.streamChat(compareMessages, {
                    onToken: (token) => { results[server.name] += token; updateStreamDisplay(); },
                    onDone: (s) => { cmpStats[server.name] = { tokPerSec: s.tokPerSec, tokens: s.totalTokens }; updateStreamDisplay(); resolve(); },
                    onError: (err) => { results[server.name] = `ERROR: ${err}`; updateStreamDisplay(); resolve(); },
                  }).catch((e) => { results[server.name] = `ERROR: ${e.message?.split("\n")[0]}`; updateStreamDisplay(); resolve(); });
                });
              } catch (e: any) {
                results[server.name] = `ERROR: ${e.message?.split("\n")[0] || "Connection failed"}`;
                updateStreamDisplay();
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
                const genResult = await client.chat([
                  { role: "system", content: "Generate a single macOS shell command for the user's request. Output ONLY the command, nothing else." },
                  { role: "user", content: args },
                ]);
                cmd = genResult.text.trim().replace(/^`+|`+$/g, "");
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
import sys, json
from ddgs import DDGS
from datetime import datetime
query = sys.argv[1]
is_news = sys.argv[2] == '1'
seen = set()
out = []

def add(title, body, date='', src='', url=''):
    key = title.strip().lower()[:60]
    if key in seen: return
    seen.add(key)
    line = ''
    if date: line += '[' + date[:10] + '] '
    line += title.strip()
    if src: line += ' (' + src + ')'
    line += ': ' + body.strip()
    if url: line += ' | ' + url
    out.append(line)

with DDGS() as d:
    if is_news:
        try:
            for r in d.news(query, max_results=8):
                add(r.get('title',''), r.get('body',''), r.get('date',''), r.get('source',''), r.get('url',''))
        except: pass
        try:
            for r in d.text(query + ' ' + datetime.now().strftime('%Y-%m-%d'), max_results=5):
                add(r.get('title',''), r.get('body',''), url=r.get('href',''))
        except: pass
    else:
        try:
            for r in d.text(query + ' ' + datetime.now().strftime('%B %Y'), max_results=8):
                add(r.get('title',''), r.get('body',''), url=r.get('href',''))
        except: pass
        if len(out) < 4:
            try:
                for r in d.news(query, max_results=5):
                    add(r.get('title',''), r.get('body',''), r.get('date',''), r.get('source',''))
            except: pass

for line in out[:10]:
    print('- ' + line)
" "${safeQuery}" "${isNews ? '1' : '0'}"`,
                { encoding: "utf-8", timeout: 20000 }
              ).trim();
              setSpinnerMsg("");
              if (searchResults) {
                // Trim results to avoid context overflow — keep first 6 results, cap each at 200 chars
                const trimmedResults = searchResults.split('\n').slice(0, 6).map(l => l.slice(0, 200)).join('\n');
                setEntries((prev) => [
                  ...prev,
                  { role: "info", content: `◆ Found ${searchResults.split('\n').length} results:\n${trimmedResults.split('\n').map(l => '  ' + l.slice(0, 120)).join('\n')}` },
                ]);
                setSpinnerMsg("Synthesizing answer...");
                // Send trimmed results to LLM — don't bloat conversation history
                const today = new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
                // Use a temporary message array to avoid polluting conversation context
                const searchMessages = [
                  messagesRef.current[0], // system prompt
                  {
                    role: "user" as const,
                    content: `Today is ${today}. Summarize these search results concisely. Include names, dates, facts.\n\nResults:\n${trimmedResults}\n\nQuestion: ${args}`,
                  },
                ];
                // Stream LLM response (falls through to streaming below)
                setIsStreaming(true);
                setStreamingText("");
                onStatusChange("streaming");
                abortRef.current = new AbortController();
                let fullText = "";
                await client.streamChat(searchMessages, {
                  onToken: (token) => { setSpinnerMsg(""); fullText += token; setStreamingText(fullText); },
                  onDone: (stats) => {
                    setSpinnerMsg("");
                    // Store only the summary in conversation history, not raw search data
                    messagesRef.current.push({ role: "user", content: `[searched: ${args}]` });
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
              const docExt = (docPath.split(".").pop() || "").toLowerCase();
              if (docExt === "docx") {
                parsed = execSync(
                  `/opt/homebrew/bin/python3.13 -c "
from docx import Document
doc = Document('${docPath.replace(/'/g, "\\'")}')
for p in doc.paragraphs:
    if p.text.strip(): print(p.text)
for t in doc.tables:
    for row in t.rows:
        print(' | '.join(c.text for c in row.cells))
" 2>&1 | head -300`,
                  { encoding: "utf-8", timeout: 15000 }
                ).trim();
              } else {
                execSync(`cp "${docPath}" "/tmp/tinybit_doc.${docExt}"`, { timeout: 5000 });
                parsed = execSync(`cd /Users/nicozahniser/mac-code-ui && npx lit parse "/tmp/tinybit_doc.${docExt}" --dpi 72 --num-workers 1 -q 2>/dev/null | head -300`, {
                  encoding: "utf-8",
                  timeout: 30000,
                }).trim();
              }
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
              { name: "Gemma 4-26B MoE (IQ2)", size: "9.3 GB", speed: "36 tok/s (16GB) / 1.4 tok/s (8GB)", repo: "unsloth/gemma-4-26B-A4B-it-GGUF", file: "gemma-4-26B-A4B-it-UD-IQ2_M.gguf", port: 8205, note: "build llama.cpp from source + --reasoning off" },
              { name: "Gemma 4-26B MoE (Q4)", size: "16.9 GB", speed: "5 tok/s (16GB)", repo: "unsloth/gemma-4-26B-A4B-it-GGUF", file: "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", port: 8205, note: "build llama.cpp from source + --reasoning off" },
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

      // ── PicoClaw-inspired: Steering Queue ──
      // If we're already in the agent loop, queue this message for injection
      if (isProcessingRef.current) {
        steeringQueueRef.current.push(trimmed);
        setEntries((prev) => [...prev, { role: "info", content: "◆ Queued (will process after current tool finishes)" }]);
        return;
      }

      isProcessingRef.current = true;

      // Update system prompt with current tool set (may have promoted tools)
      messagesRef.current[0] = { role: "system", content: buildSystemPrompt() };

      // ── PicoClaw-inspired: Tool TTL — tick down promoted tools ──
      promotedToolsRef.current = promotedToolsRef.current
        .map(t => ({ ...t, ttl: (t.ttl || 1) - 1 }))
        .filter(t => (t.ttl || 0) > 0);

      // Agent loop: call LLM, check for tool calls, execute, feed back, repeat
      let loopCount = 0;
      const MAX_LOOPS = 8; // increased — parallel tool calls count as 1 loop

      while (loopCount < MAX_LOOPS) {
        loopCount++;
        setSpinnerMsg(loopCount === 1 ? "Thinking..." : "Processing tool results...");

        // ── PicoClaw-inspired: Context Budget — trim before every LLM call ──
        messagesRef.current = trimToContextBudget(messagesRef.current, CONTEXT_BUDGET);

        // Get non-streaming response to check for tool calls
        let response = "";
        try {
          const chatResult = await client.chat(messagesRef.current, 500);
          response = chatResult.text;
          // Report stats from agent loop
          onStatsUpdate({ tokPerSec: chatResult.tokPerSec, totalTokens: chatResult.totalTokens });
        } catch (e: any) {
          setSpinnerMsg("");
          setEntries((prev) => [...prev, { role: "error", content: `◆ Error: ${e.message}` }]);
          break;
        }

        setSpinnerMsg("");

        // ── PicoClaw-inspired: Parallel Tool Execution ──
        // Extract ALL tool_call tags (not just the first)
        const toolCallMatches = [...response.matchAll(/<tool_call>([\s\S]*?)<\/tool_call>/g)];

        if (toolCallMatches.length === 0) {
          // No tool call — final text response
          messagesRef.current.push({ role: "assistant", content: response });
          setEntries((prev) => [...prev, { role: "assistant", content: response }]);
          break;
        }

        // Show what the model is doing
        messagesRef.current.push({ role: "assistant", content: response });

        // ── PicoClaw-inspired: Steering Queue check ──
        // If user sent a message while we were processing, inject it
        if (steeringQueueRef.current.length > 0) {
          const queued = steeringQueueRef.current.shift()!;
          setEntries((prev) => [...prev, { role: "info", content: `◆ Interrupting: processing queued message` }]);
          messagesRef.current.push({ role: "user", content: queued });
          continue; // skip remaining tool calls, process new message
        }

        // Parse all tool calls
        const toolCalls: Array<{ name: string; arguments: any }> = [];
        for (const match of toolCallMatches) {
          try {
            toolCalls.push(JSON.parse(match[1]));
          } catch (e: any) {
            setEntries((prev) => [...prev, { role: "error", content: `◆ Tool parse error: ${e.message}` }]);
          }
        }

        if (toolCalls.length === 0) {
          const cleanResponse = response.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();
          if (cleanResponse) {
            setEntries((prev) => [...prev, { role: "assistant", content: cleanResponse }]);
          }
          break;
        }

        // Execute tool calls — parallel when multiple, sequential when single
        const executeTool = async (toolCall: { name: string; arguments: any }): Promise<string> => {
          const toolName = toolCall.name;
          const toolArgs = toolCall.arguments || {};

          // ── search_tools: PicoClaw Tool Discovery ──
          if (toolName === "search_tools") {
            const query = (toolArgs.query || "").toLowerCase();
            const matches = HIDDEN_TOOLS.filter(t =>
              t.name.includes(query) || t.description.toLowerCase().includes(query)
            );
            if (matches.length > 0) {
              // Promote matched tools with TTL
              for (const tool of matches) {
                const existing = promotedToolsRef.current.find(t => t.name === tool.name);
                if (existing) {
                  existing.ttl = TOOL_PROMOTE_TTL;
                } else {
                  promotedToolsRef.current.push({ ...tool, hidden: false, ttl: TOOL_PROMOTE_TTL });
                }
              }
              // Rebuild system prompt with newly promoted tools
              messagesRef.current[0] = { role: "system", content: buildSystemPrompt() };
              const found = matches.map(t => `${t.name}: ${t.description}`).join('\n');
              setEntries((prev) => [...prev, { role: "info", content: `◆ Found ${matches.length} tools:\n${found}` }]);
              return JSON.stringify({ tools_found: matches.map(t => t.name), message: "These tools are now available. Use them in your next response." });
            }
            return JSON.stringify({ tools_found: [], message: "No matching tools found. Use shell for general tasks." });
          }

          // ── write_file (hidden, promoted via search_tools) ──
          if (toolName === "write_file") {
            const filePath = (toolArgs.path || "").replace(/^~/, process.env.HOME || "");
            try {
              const { writeFileSync } = await import("node:fs");
              writeFileSync(filePath, toolArgs.content || "");
              setEntries((prev) => [...prev, { role: "info", content: `◆ Wrote: ${filePath}` }]);
              return `File written successfully: ${filePath}`;
            } catch (e: any) {
              return `Error writing file: ${e.message}`;
            }
          }

          // ── list_dir (hidden, promoted via search_tools) ──
          if (toolName === "list_dir") {
            const dirPath = (toolArgs.path || "~/").replace(/^~/, process.env.HOME || "");
            try {
              const output = execSync(`ls -la "${dirPath}" | head -30`, { encoding: "utf-8", timeout: 5000 }).trim();
              setEntries((prev) => [...prev, { role: "info", content: `◆ Directory: ${dirPath}\n${output.split('\n').slice(0, 15).join('\n')}` }]);
              return output;
            } catch (e: any) {
              return `Error: ${e.message?.split("\n")[0]}`;
            }
          }

          // ── screenshot (hidden, promoted via search_tools) ──
          if (toolName === "screenshot") {
            try {
              execSync("screencapture -x /tmp/tinybit-screenshot.png", { timeout: 5000 });
              setEntries((prev) => [...prev, { role: "info", content: "◆ Screenshot saved to /tmp/tinybit-screenshot.png" }]);
              return "Screenshot captured at /tmp/tinybit-screenshot.png";
            } catch {
              return "Screenshot failed";
            }
          }

          // ── system_info (hidden, promoted via search_tools) ──
          if (toolName === "system_info") {
            try {
              const info = execSync(`echo "OS: $(sw_vers -productName) $(sw_vers -productVersion)"; echo "Chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo Apple Silicon)"; echo "RAM: $(sysctl -n hw.memsize | awk '{print $1/1073741824}') GB"; echo "Disk: $(df -h / | tail -1 | awk '{print $4 " free of " $2}')"`, { encoding: "utf-8", timeout: 5000 }).trim();
              setEntries((prev) => [...prev, { role: "info", content: `◆ System Info:\n${info}` }]);
              return info;
            } catch (e: any) {
              return `Error: ${e.message?.split("\n")[0]}`;
            }
          }

          // ── shell ──
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
            return truncOutput;
          }

          // ── web_search ──
          if (toolName === "web_search") {
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
seen = set()
out = []

def add(title, body, date='', src='', url=''):
    key = title.strip().lower()[:60]
    if key in seen: return
    seen.add(key)
    line = ''
    if date: line += '[' + date[:10] + '] '
    line += title.strip()
    if src: line += ' (' + src + ')'
    line += ': ' + body.strip()
    out.append(line)

with DDGS() as d:
    if is_news:
        try:
            for r in d.news(q, max_results=8):
                add(r.get('title',''), r.get('body',''), r.get('date',''), r.get('source',''), r.get('url',''))
        except: pass
        try:
            for r in d.text(q + ' ' + datetime.now().strftime('%Y-%m-%d'), max_results=5):
                add(r.get('title',''), r.get('body',''), url=r.get('href',''))
        except: pass
    else:
        try:
            for r in d.text(q + ' ' + datetime.now().strftime('%B %Y'), max_results=8):
                add(r.get('title',''), r.get('body',''), url=r.get('href',''))
        except: pass
        if len(out) < 4:
            try:
                for r in d.news(q, max_results=5):
                    add(r.get('title',''), r.get('body',''), r.get('date',''), r.get('source',''))
            except: pass

for line in out[:10]:
    print('- ' + line)
" "${safeQuery}" "${isNews ? '1' : '0'}"`,
                { encoding: "utf-8", timeout: 20000 }
              ).trim();
            } catch {
              results = "Search failed — try different keywords";
            }
            setSpinnerMsg("");
            const trimmed = results.split('\n').slice(0, 6).map((l: string) => l.slice(0, 200)).join('\n');
            setEntries((prev) => [...prev, { role: "info", content: `◆ Results:\n${trimmed.split('\n').map((l: string) => '  ' + l.slice(0, 120)).join('\n')}` }]);
            return trimmed;
          }

          // ── read_file ──
          if (toolName === "read_file") {
            let filePath = (toolArgs.path || "").replace(/^~/, process.env.HOME || "");
            if (!filePath.startsWith("/")) {
              const home = process.env.HOME || "";
              for (const dir of ["Desktop", "Downloads", "Documents", ""]) {
                const candidate = dir ? `${home}/${dir}/${filePath}` : `${home}/${filePath}`;
                try { execSync(`test -f "${candidate}"`, { timeout: 1000 }); filePath = candidate; break; } catch {}
              }
            }
            const isDocument = /\.(pdf|docx|xlsx|pptx|doc|png|jpg|jpeg|tiff|bmp|heic)$/i.test(filePath);
            setSpinnerMsg(isDocument ? `Parsing: ${filePath}` : `Reading: ${filePath}`);
            setEntries((prev) => [...prev, { role: "info", content: `◆ ${isDocument ? "Parsing" : "Reading"}: ${filePath}` }]);
            let content = "";
            try {
              if (isDocument) {
                const ext = (filePath.split(".").pop() || "").toLowerCase();
                if (ext === "docx") {
                  content = execSync(`/opt/homebrew/bin/python3.13 -c "
from docx import Document
doc = Document('${filePath.replace(/'/g, "\\'")}')
for p in doc.paragraphs:
    if p.text.strip(): print(p.text)
for t in doc.tables:
    for row in t.rows:
        print(' | '.join(c.text for c in row.cells))
" 2>&1 | head -200`, { encoding: "utf-8", timeout: 15000 }).trim();
                } else {
                  execSync(`cp "${filePath}" "/tmp/tinybit_parse.${ext}"`, { timeout: 5000 });
                  content = execSync(`cd /Users/nicozahniser/mac-code-ui && npx lit parse "/tmp/tinybit_parse.${ext}" --dpi 72 --num-workers 1 -q 2>/dev/null | head -200`, { encoding: "utf-8", timeout: 30000 }).trim();
                }
                if (!content || content.length < 10) content = "Document returned no content.";
              } else {
                content = execSync(`head -80 "${filePath}"`, { encoding: "utf-8", timeout: 5000 }).trim();
              }
            } catch (err: any) {
              content = `Error: ${err.message?.split("\n")[0]}`;
            }
            setSpinnerMsg("");
            return content;
          }

          return `Unknown tool: ${toolName}. Use search_tools to find available tools.`;
        };

        // ── Execute tools — parallel when multiple ──
        try {
          const toolResults: string[] = [];
          if (toolCalls.length === 1) {
            // Single tool — execute directly
            toolResults.push(await executeTool(toolCalls[0]));
          } else {
            // Multiple tools — execute in parallel (PicoClaw pattern)
            setEntries((prev) => [...prev, { role: "info", content: `◆ Running ${toolCalls.length} tools in parallel...` }]);
            const results = await Promise.all(toolCalls.map(tc => {
              // Wrap each in try/catch so one failure doesn't kill the batch (PicoClaw panic recovery)
              return executeTool(tc).catch(e => `Error: ${e.message}`);
            }));
            toolResults.push(...results);
          }

          // Feed all results back as one message
          const combinedResults = toolCalls.map((tc, i) =>
            `<tool_response>{"name": "${tc.name}", "output": ${JSON.stringify(toolResults[i] || "No output")}}</tool_response>`
          ).join('\n');
          messagesRef.current.push({ role: "user", content: combinedResults });

        } catch (e: any) {
          setEntries((prev) => [...prev, { role: "error", content: `◆ Tool error: ${e.message}` }]);
          const cleanResponse = response.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();
          if (cleanResponse) {
            setEntries((prev) => [...prev, { role: "assistant", content: cleanResponse }]);
          }
          break;
        }
      }

      if (loopCount >= MAX_LOOPS) {
        setEntries((prev) => [...prev, { role: "info", content: "◆ Reached max tool iterations." }]);
      }

      isProcessingRef.current = false;

      // ── PicoClaw-inspired: Drain steering queue ──
      if (steeringQueueRef.current.length > 0) {
        const next = steeringQueueRef.current.shift()!;
        // Process next queued message
        handleSubmit(next);
        return;
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
