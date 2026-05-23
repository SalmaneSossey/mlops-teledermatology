import { StyleSheet, Text, View } from "react-native";

import { colors, radii, riskColor, spacing, statusColor } from "../theme";

type BadgeProps = {
  label: string;
  tone: "risk" | "status";
};

export function Badge({ label, tone }: BadgeProps) {
  const color = tone === "risk" ? riskColor(label) : statusColor(label);
  return (
    <View style={[styles.badge, { borderColor: color, backgroundColor: `${color}14` }]}>
      <Text style={[styles.text, { color }]}>{label.toUpperCase()}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    alignSelf: "flex-start",
    borderRadius: radii.sm,
    borderWidth: 1,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs
  },
  text: {
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0
  }
});
