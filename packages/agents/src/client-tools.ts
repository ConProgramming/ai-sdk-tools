/**
 * Client-side tool utilities for AI SDK Tools
 *
 * Client-side tools are tools that execute in the browser rather than on the server.
 * They're useful for:
 * - Accessing browser APIs (clipboard, notifications, geolocation, etc.)
 * - Interacting with client-side state (React state, Zustand stores)
 * - Reducing server load for lightweight operations
 * - Privacy-sensitive operations that shouldn't leave the client
 *
 * @example
 * ```typescript
 * // Define a client-side tool (no execute function)
 * const copyToClipboard = clientTool({
 *   description: 'Copy text to the clipboard',
 *   inputSchema: z.object({
 *     text: z.string().describe('The text to copy'),
 *   }),
 * });
 *
 * // Use in agent
 * const agent = new Agent({
 *   name: 'assistant',
 *   tools: { copyToClipboard },
 * });
 *
 * // Handle on client with useClientTools hook
 * const { pendingToolCalls, submitToolResult } = useClientTools({
 *   handlers: {
 *     copyToClipboard: async ({ text }) => {
 *       await navigator.clipboard.writeText(text);
 *       return { success: true };
 *     },
 *   },
 * });
 * ```
 */

import { tool, type Tool } from "ai";
import type { z } from "zod";

/**
 * Configuration for a client-side tool
 */
export interface ClientToolConfig<TInput extends z.ZodType = z.ZodType> {
  /** Description of what the tool does */
  description: string;
  /** Zod schema for the tool input */
  inputSchema: TInput;
}

/**
 * A tool call that is pending client-side execution
 */
export interface PendingClientToolCall<TArgs = unknown> {
  /** Unique identifier for this tool call */
  toolCallId: string;
  /** Name of the tool being called */
  toolName: string;
  /** Arguments passed to the tool */
  args: TArgs;
  /** Timestamp when the tool call was received */
  timestamp: number;
}

/**
 * Result of a client-side tool execution
 */
export interface ClientToolResult<TResult = unknown> {
  /** The tool call ID this result is for */
  toolCallId: string;
  /** The result of the tool execution */
  result: TResult;
}

/**
 * Create a client-side tool definition.
 *
 * Client-side tools don't have an execute function - they're executed
 * in the browser using the useClientTools hook.
 *
 * @param config - Tool configuration
 * @returns A tool definition compatible with AI SDK
 *
 * @example
 * ```typescript
 * import { clientTool } from '@ai-sdk-tools/agents';
 * import { z } from 'zod';
 *
 * const showNotification = clientTool({
 *   description: 'Show a browser notification',
 *   inputSchema: z.object({
 *     title: z.string(),
 *     body: z.string().optional(),
 *   }),
 * });
 * ```
 */
export function clientTool<TInput extends z.ZodType>(
  config: ClientToolConfig<TInput>,
): Tool<z.infer<TInput>, never> {
  // Create a tool without an execute function
  // This signals to the AI SDK that the tool should be handled client-side
  return tool({
    description: config.description,
    inputSchema: config.inputSchema,
    // No execute function - this is the key indicator for client-side tools
  }) as Tool<z.infer<TInput>, never>;
}

/**
 * Check if a tool is a client-side tool (has no execute function)
 *
 * @param toolDef - The tool definition to check
 * @returns True if the tool is client-side (no execute function)
 */
export function isClientTool(toolDef: Tool): boolean {
  // Client-side tools don't have an execute function
  return (
    toolDef !== null &&
    typeof toolDef === "object" &&
    !("execute" in toolDef && typeof (toolDef as any).execute === "function")
  );
}

/**
 * Get the names of all client-side tools from a tools object
 *
 * @param tools - Object containing tool definitions
 * @returns Array of tool names that are client-side
 */
export function getClientToolNames(tools: Record<string, Tool>): string[] {
  return Object.entries(tools)
    .filter(([_, toolDef]) => isClientTool(toolDef))
    .map(([name]) => name);
}

/**
 * Type helper for extracting the input type from a client tool
 */
export type ClientToolInput<T extends Tool> = T extends Tool<infer TInput, any>
  ? TInput
  : never;

/**
 * Type helper for client tool handlers
 */
export type ClientToolHandler<TArgs = unknown, TResult = unknown> = (
  args: TArgs,
) => TResult | Promise<TResult>;

/**
 * Map of client tool handlers
 */
export type ClientToolHandlers<TTools extends Record<string, Tool>> = {
  [K in keyof TTools]?: ClientToolHandler<ClientToolInput<TTools[K]>>;
};
