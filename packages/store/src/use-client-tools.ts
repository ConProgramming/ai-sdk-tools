/**
 * useClientTools - React hook for handling client-side tool execution
 *
 * This hook provides a seamless way to handle AI tool calls that should
 * execute in the browser rather than on the server.
 *
 * @example
 * ```typescript
 * import { useChat, useClientTools } from '@ai-sdk-tools/store';
 *
 * function ChatComponent() {
 *   const { messages, sendMessage, addToolResult } = useChat({
 *     api: '/api/chat',
 *   });
 *
 *   // Handle client-side tools
 *   useClientTools({
 *     messages,
 *     addToolResult,
 *     handlers: {
 *       copyToClipboard: async ({ text }) => {
 *         await navigator.clipboard.writeText(text);
 *         return { success: true };
 *       },
 *       showNotification: async ({ title, body }) => {
 *         new Notification(title, { body });
 *         return { shown: true };
 *       },
 *     },
 *   });
 *
 *   return <div>...</div>;
 * }
 * ```
 */

import type { UIMessage } from "@ai-sdk/react";
import { useCallback, useEffect, useRef } from "react";

/**
 * A pending tool call that needs client-side execution
 */
export interface PendingToolCall {
  /** Unique identifier for this tool call */
  toolCallId: string;
  /** Name of the tool being called */
  toolName: string;
  /** Arguments passed to the tool */
  args: unknown;
  /** The message ID containing this tool call */
  messageId: string;
}

/**
 * Handler function for a client-side tool
 */
export type ToolHandler<TArgs = unknown, TResult = unknown> = (
  args: TArgs,
) => TResult | Promise<TResult>;

/**
 * Map of tool handlers by tool name
 */
export type ToolHandlers = Record<string, ToolHandler>;

/**
 * Options for the useClientTools hook
 */
export interface UseClientToolsOptions<TMessage extends UIMessage = UIMessage> {
  /** Current chat messages */
  messages: TMessage[];
  /**
   * Function to submit a tool result back to the AI
   * This comes from useChat's addToolResult
   */
  addToolResult: (result: {
    toolCallId: string;
    result: unknown;
  }) => void;
  /**
   * Map of tool handlers by tool name
   * Tools not in this map will remain pending
   */
  handlers: ToolHandlers;
  /**
   * Optional callback when a tool execution fails
   */
  onError?: (error: Error, toolCall: PendingToolCall) => void;
  /**
   * Optional callback when a tool execution completes
   */
  onComplete?: (toolCall: PendingToolCall, result: unknown) => void;
  /**
   * If true, automatically execute handlers when tool calls are detected
   * Default: true
   */
  autoExecute?: boolean;
}

/**
 * Return value of the useClientTools hook
 */
export interface UseClientToolsReturn {
  /** Tool calls that are waiting for execution (no handler provided) */
  pendingToolCalls: PendingToolCall[];
  /** Manually execute a pending tool call with a result */
  submitResult: (toolCallId: string, result: unknown) => void;
  /** Check if a specific tool call is being processed */
  isProcessing: (toolCallId: string) => boolean;
}

/**
 * Extract pending tool calls from messages
 *
 * A tool call is pending when:
 * - state is 'input-available' (input ready, no output yet)
 * - state is 'input-streaming' (input still being streamed - we'll wait)
 *
 * We only process 'input-available' state where the full input is ready.
 */
function extractPendingToolCalls<TMessage extends UIMessage>(
  messages: TMessage[],
  processedIds: Set<string>,
): PendingToolCall[] {
  const pending: PendingToolCall[] = [];

  for (const message of messages) {
    // Only check assistant messages
    if (message.role !== "assistant") continue;

    // Check for tool invocations in the message parts
    if (Array.isArray(message.parts)) {
      for (const part of message.parts) {
        // Check for tool-invocation type
        if (part.type === "tool-invocation") {
          const toolPart = part as {
            type: "tool-invocation";
            toolCallId: string;
            toolName: string;
            input: unknown;
            state:
              | "input-streaming"
              | "input-available"
              | "output-available"
              | "output-error";
          };

          // Only process tool calls where input is available but no output yet
          if (toolPart.state !== "input-available") continue;

          // Skip if already processed
          if (processedIds.has(toolPart.toolCallId)) continue;

          pending.push({
            toolCallId: toolPart.toolCallId,
            toolName: toolPart.toolName,
            args: toolPart.input,
            messageId: message.id,
          });
        }
      }
    }
  }

  return pending;
}

/**
 * React hook for handling client-side tool execution
 *
 * This hook automatically detects tool calls in chat messages and either:
 * 1. Executes the provided handler automatically (if autoExecute is true)
 * 2. Returns them as pending for manual handling
 *
 * @param options - Configuration options
 * @returns Object with pending tool calls and helper functions
 */
export function useClientTools<TMessage extends UIMessage = UIMessage>(
  options: UseClientToolsOptions<TMessage>,
): UseClientToolsReturn {
  const {
    messages,
    addToolResult,
    handlers,
    onError,
    onComplete,
    autoExecute = true,
  } = options;

  // Track which tool calls have been processed to avoid double execution
  const processedRef = useRef<Set<string>>(new Set());
  // Track which tool calls are currently being processed
  const processingRef = useRef<Set<string>>(new Set());

  // Extract pending tool calls
  const pendingToolCalls = extractPendingToolCalls(
    messages,
    processedRef.current,
  );

  // Submit a result for a tool call
  const submitResult = useCallback(
    (toolCallId: string, result: unknown) => {
      // Mark as processed
      processedRef.current.add(toolCallId);
      processingRef.current.delete(toolCallId);

      // Submit to the AI
      addToolResult({
        toolCallId,
        result,
      });
    },
    [addToolResult],
  );

  // Check if a tool call is being processed
  const isProcessing = useCallback((toolCallId: string) => {
    return processingRef.current.has(toolCallId);
  }, []);

  // Auto-execute handlers for pending tool calls
  useEffect(() => {
    if (!autoExecute) return;

    for (const toolCall of pendingToolCalls) {
      const { toolCallId, toolName, args } = toolCall;

      // Skip if already processing or processed
      if (
        processingRef.current.has(toolCallId) ||
        processedRef.current.has(toolCallId)
      ) {
        continue;
      }

      // Check if we have a handler for this tool
      const handler = handlers[toolName];
      if (!handler) continue;

      // Mark as processing
      processingRef.current.add(toolCallId);

      // Execute the handler
      Promise.resolve()
        .then(() => handler(args))
        .then((result) => {
          submitResult(toolCallId, result);
          onComplete?.(toolCall, result);
        })
        .catch((error) => {
          // Mark as processed even on error to avoid retry loops
          processedRef.current.add(toolCallId);
          processingRef.current.delete(toolCallId);

          // Submit error result
          addToolResult({
            toolCallId,
            result: {
              error: error instanceof Error ? error.message : String(error),
            },
          });

          onError?.(
            error instanceof Error ? error : new Error(String(error)),
            toolCall,
          );
        });
    }
  }, [
    pendingToolCalls,
    handlers,
    autoExecute,
    submitResult,
    addToolResult,
    onError,
    onComplete,
  ]);

  // Filter out tool calls that have handlers (they'll be auto-executed)
  const unhandledToolCalls = pendingToolCalls.filter(
    (tc) => !handlers[tc.toolName],
  );

  return {
    pendingToolCalls: unhandledToolCalls,
    submitResult,
    isProcessing,
  };
}
