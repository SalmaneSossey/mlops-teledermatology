import { StyleSheet, Text, View } from "react-native";

import { colors, radii, spacing } from "../theme";

type ProbabilityBarsProps = {
  probabilities: Record<string, number>;
};

export function ProbabilityBars({ probabilities }: ProbabilityBarsProps) {
  const rows = Object.entries(probabilities).sort((left, right) => right[1] - left[1]);
  return (
    <View style={styles.container}>
      {rows.map(([label, value]) => {
        const percentage = Math.round(value * 100);
        return (
          <View key={label} style={styles.row}>
            <View style={styles.rowHeader}>
              <Text style={styles.label}>{label}</Text>
              <Text style={styles.value}>{percentage}%</Text>
            </View>
            <View style={styles.track}>
              <View style={[styles.fill, { width: `${Math.max(2, percentage)}%` }]} />
            </View>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: spacing.md
  },
  row: {
    gap: spacing.xs
  },
  rowHeader: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between"
  },
  label: {
    color: colors.text,
    fontSize: 14,
    fontWeight: "700"
  },
  value: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: "600"
  },
  track: {
    backgroundColor: colors.surfaceMuted,
    borderRadius: radii.sm,
    height: 10,
    overflow: "hidden"
  },
  fill: {
    backgroundColor: colors.primary,
    borderRadius: radii.sm,
    height: "100%"
  }
});
