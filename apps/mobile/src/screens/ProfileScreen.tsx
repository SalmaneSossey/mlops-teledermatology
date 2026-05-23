import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View
} from "react-native";

import { API_BASE_URL, getCurrentModel } from "../api";
import { colors, radii, spacing } from "../theme";
import { ModelCurrentResponse } from "../types";

type ProfileScreenProps = {
  token: string;
  onLogout: () => Promise<void>;
};

export function ProfileScreen({ token, onLogout }: ProfileScreenProps) {
  const [model, setModel] = useState<ModelCurrentResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadModel() {
      setLoading(true);
      setError(null);
      try {
        setModel(await getCurrentModel(token));
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : "Model status unavailable");
      } finally {
        setLoading(false);
      }
    }
    void loadModel();
  }, [token]);

  return (
    <ScrollView contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Text style={styles.title}>Profile</Text>
        <Text style={styles.subtitle}>patient@example.com</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Connection</Text>
        <Text style={styles.value}>{API_BASE_URL}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Active Model</Text>
        {loading ? <ActivityIndicator color={colors.primary} /> : null}
        {error ? <Text style={styles.error}>{error}</Text> : null}
        {model ? (
          <>
            <Text style={styles.value}>{model.available ? model.model_run_id : "Unavailable"}</Text>
            <Text style={styles.muted}>{model.warning}</Text>
          </>
        ) : null}
      </View>

      <Pressable onPress={onLogout} style={styles.logoutButton}>
        <Text style={styles.logoutText}>Log out</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  content: {
    gap: spacing.lg,
    padding: spacing.lg,
    paddingBottom: 110
  },
  header: {
    gap: spacing.xs
  },
  title: {
    color: colors.text,
    fontSize: 28,
    fontWeight: "900",
    letterSpacing: 0
  },
  subtitle: {
    color: colors.muted,
    fontSize: 15
  },
  card: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.md,
    borderWidth: 1,
    gap: spacing.md,
    padding: spacing.lg
  },
  cardTitle: {
    color: colors.text,
    fontSize: 17,
    fontWeight: "900"
  },
  value: {
    color: colors.text,
    fontSize: 15,
    fontWeight: "700",
    lineHeight: 21
  },
  muted: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 20
  },
  logoutButton: {
    alignItems: "center",
    borderColor: colors.danger,
    borderRadius: radii.sm,
    borderWidth: 1,
    justifyContent: "center",
    minHeight: 50
  },
  logoutText: {
    color: colors.danger,
    fontSize: 16,
    fontWeight: "900"
  },
  error: {
    color: colors.danger,
    fontSize: 14,
    fontWeight: "700"
  }
});
