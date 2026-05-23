import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View
} from "react-native";

import { listPatientHistory } from "../api";
import { ProbabilityBars } from "../components/ProbabilityBars";
import { Badge } from "../components/RiskBadge";
import { colors, radii, spacing } from "../theme";
import { PatientConsultationHistoryResponse } from "../types";

type HistoryScreenProps = {
  token: string;
  refreshKey: number;
};

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

function metadataSummary(metadata: Record<string, unknown>): string {
  const region = metadata.region ? String(metadata.region) : "UNK";
  const age = metadata.age ? String(metadata.age) : "UNK";
  return `Age ${age} - ${region}`;
}

export function HistoryScreen({ token, refreshKey }: HistoryScreenProps) {
  const [items, setItems] = useState<PatientConsultationHistoryResponse[]>([]);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadHistory() {
    setLoading(true);
    setError(null);
    try {
      setItems(await listPatientHistory(token));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not load history");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadHistory();
  }, [refreshKey]);

  return (
    <ScrollView contentContainerStyle={styles.content}>
      <View style={styles.headerRow}>
        <View>
          <Text style={styles.title}>History</Text>
          <Text style={styles.subtitle}>Submitted patient cases</Text>
        </View>
        <Pressable onPress={loadHistory} style={styles.refreshButton}>
          <Text style={styles.refreshText}>Refresh</Text>
        </Pressable>
      </View>
      {loading ? <ActivityIndicator color={colors.primary} /> : null}
      {error ? <Text style={styles.error}>{error}</Text> : null}
      {!loading && items.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyTitle}>No cases yet</Text>
        </View>
      ) : null}
      {items.map((item) => {
        const { consultation, latest_prediction: prediction } = item;
        const expanded = expandedId === consultation.id;
        return (
          <Pressable
            key={consultation.id}
            onPress={() => setExpandedId(expanded ? null : consultation.id)}
            style={styles.card}
          >
            <View style={styles.cardHeader}>
              <View style={styles.cardTitleGroup}>
                <Text style={styles.caseTitle}>Case {consultation.id}</Text>
                <Text style={styles.meta}>{formatDate(consultation.created_at)}</Text>
              </View>
              <Badge label={consultation.status} tone="status" />
            </View>
            <Text style={styles.meta}>{metadataSummary(consultation.clinical_metadata)}</Text>
            {consultation.symptoms_notes ? (
              <Text numberOfLines={expanded ? undefined : 2} style={styles.notes}>
                {consultation.symptoms_notes}
              </Text>
            ) : null}
            {prediction ? (
              <View style={styles.predictionStrip}>
                <Badge label={prediction.risk_level} tone="risk" />
                <Text style={styles.predictionText}>{prediction.predicted_label}</Text>
              </View>
            ) : (
              <Text style={styles.pending}>No prediction yet</Text>
            )}
            {expanded && prediction ? (
              <View style={styles.expanded}>
                <ProbabilityBars probabilities={prediction.probabilities} />
                <Text style={styles.warning}>{prediction.warning}</Text>
              </View>
            ) : null}
          </Pressable>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  content: {
    gap: spacing.lg,
    padding: spacing.lg,
    paddingBottom: 110
  },
  headerRow: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between"
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
  refreshButton: {
    borderColor: colors.primary,
    borderRadius: radii.sm,
    borderWidth: 1,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm
  },
  refreshText: {
    color: colors.primaryDark,
    fontSize: 14,
    fontWeight: "800"
  },
  card: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.md,
    borderWidth: 1,
    gap: spacing.md,
    padding: spacing.lg
  },
  cardHeader: {
    alignItems: "flex-start",
    flexDirection: "row",
    gap: spacing.md,
    justifyContent: "space-between"
  },
  cardTitleGroup: {
    flex: 1,
    gap: spacing.xs
  },
  caseTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: "900"
  },
  meta: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: "600"
  },
  notes: {
    color: colors.text,
    fontSize: 15,
    lineHeight: 21
  },
  predictionStrip: {
    alignItems: "center",
    flexDirection: "row",
    gap: spacing.md
  },
  predictionText: {
    color: colors.text,
    fontSize: 20,
    fontWeight: "900"
  },
  pending: {
    color: colors.inactive,
    fontSize: 14,
    fontWeight: "700"
  },
  expanded: {
    borderTopColor: colors.border,
    borderTopWidth: 1,
    gap: spacing.lg,
    paddingTop: spacing.md
  },
  warning: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 18
  },
  empty: {
    alignItems: "center",
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.md,
    borderWidth: 1,
    padding: spacing.xl
  },
  emptyTitle: {
    color: colors.muted,
    fontSize: 16,
    fontWeight: "700"
  },
  error: {
    color: colors.danger,
    fontSize: 14,
    fontWeight: "700"
  }
});
