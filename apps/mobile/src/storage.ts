import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

const memoryFallback = new Map<string, string>();

function canUseLocalStorage(): boolean {
  return typeof globalThis !== "undefined" && "localStorage" in globalThis;
}

export async function saveValue(key: string, value: string): Promise<void> {
  if (Platform.OS === "web" && canUseLocalStorage()) {
    globalThis.localStorage.setItem(key, value);
    return;
  }
  try {
    await SecureStore.setItemAsync(key, value);
  } catch {
    memoryFallback.set(key, value);
  }
}

export async function loadValue(key: string): Promise<string | null> {
  if (Platform.OS === "web" && canUseLocalStorage()) {
    return globalThis.localStorage.getItem(key);
  }
  try {
    const value = await SecureStore.getItemAsync(key);
    return value ?? memoryFallback.get(key) ?? null;
  } catch {
    return memoryFallback.get(key) ?? null;
  }
}

export async function deleteValue(key: string): Promise<void> {
  if (Platform.OS === "web" && canUseLocalStorage()) {
    globalThis.localStorage.removeItem(key);
    return;
  }
  try {
    await SecureStore.deleteItemAsync(key);
  } finally {
    memoryFallback.delete(key);
  }
}
