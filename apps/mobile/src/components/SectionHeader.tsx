import { StyleSheet, Text, View } from "react-native";

import { colors, spacing } from "../theme";

type SectionHeaderProps = {
  title: string;
  detail?: string;
};

export function SectionHeader({ title, detail }: SectionHeaderProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>{title}</Text>
      {detail ? <Text style={styles.detail}>{detail}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: spacing.xs,
    marginTop: spacing.lg
  },
  title: {
    color: colors.text,
    fontSize: 17,
    fontWeight: "800"
  },
  detail: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 18
  }
});
