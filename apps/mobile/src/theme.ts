export const colors = {
  background: "#f7faf9",
  surface: "#ffffff",
  surfaceMuted: "#edf7f6",
  border: "#d9e8e5",
  text: "#16302f",
  muted: "#647b78",
  primary: "#087f8c",
  primaryDark: "#075e68",
  blue: "#2563eb",
  success: "#138a55",
  warning: "#b7791f",
  danger: "#c2413a",
  inactive: "#8aa09d"
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32
};

export const radii = {
  sm: 6,
  md: 8
};

export function riskColor(riskLevel: string): string {
  const normalized = riskLevel.toLowerCase();
  if (normalized === "high") {
    return colors.danger;
  }
  if (normalized === "medium" || normalized === "moderate") {
    return colors.warning;
  }
  return colors.success;
}

export function statusColor(status: string): string {
  const normalized = status.toLowerCase();
  if (normalized === "predicted" || normalized === "reviewed") {
    return colors.primary;
  }
  if (normalized === "submitted") {
    return colors.blue;
  }
  return colors.inactive;
}
