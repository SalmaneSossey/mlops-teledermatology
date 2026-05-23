import { useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View
} from "react-native";

import { login } from "../api";
import { colors, radii, spacing } from "../theme";

type LoginScreenProps = {
  onLogin: (token: string, role: string) => Promise<void>;
};

export function LoginScreen({ onLogin }: LoginScreenProps) {
  const [email, setEmail] = useState("patient@example.com");
  const [password, setPassword] = useState("patient123");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit() {
    setLoading(true);
    setError(null);
    try {
      const payload = await login(email.trim(), password);
      if (payload.role !== "patient") {
        throw new Error("Use a patient account for the mobile app.");
      }
      await onLogin(payload.access_token, payload.role);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      style={styles.screen}
    >
      <View style={styles.header}>
        <Text style={styles.brand}>Telederm Patient</Text>
        <Text style={styles.subtitle}>Secure lesion triage companion</Text>
      </View>
      <View style={styles.form}>
        <Text style={styles.label}>Email</Text>
        <TextInput
          autoCapitalize="none"
          keyboardType="email-address"
          onChangeText={setEmail}
          style={styles.input}
          value={email}
        />
        <Text style={styles.label}>Password</Text>
        <TextInput
          onChangeText={setPassword}
          secureTextEntry
          style={styles.input}
          value={password}
        />
        {error ? <Text style={styles.error}>{error}</Text> : null}
        <Pressable disabled={loading} onPress={submit} style={styles.button}>
          {loading ? (
            <ActivityIndicator color={colors.surface} />
          ) : (
            <Text style={styles.buttonText}>Log in</Text>
          )}
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  screen: {
    backgroundColor: colors.background,
    flex: 1,
    justifyContent: "center",
    padding: spacing.xl
  },
  header: {
    gap: spacing.sm,
    marginBottom: spacing.xxl
  },
  brand: {
    color: colors.text,
    fontSize: 32,
    fontWeight: "900",
    letterSpacing: 0
  },
  subtitle: {
    color: colors.muted,
    fontSize: 16
  },
  form: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.md,
    borderWidth: 1,
    gap: spacing.md,
    padding: spacing.lg
  },
  label: {
    color: colors.text,
    fontSize: 14,
    fontWeight: "700"
  },
  input: {
    borderColor: colors.border,
    borderRadius: radii.sm,
    borderWidth: 1,
    color: colors.text,
    fontSize: 16,
    minHeight: 46,
    paddingHorizontal: spacing.md
  },
  button: {
    alignItems: "center",
    backgroundColor: colors.primary,
    borderRadius: radii.sm,
    justifyContent: "center",
    minHeight: 48
  },
  buttonText: {
    color: colors.surface,
    fontSize: 16,
    fontWeight: "800"
  },
  error: {
    color: colors.danger,
    fontSize: 14,
    fontWeight: "600"
  }
});
