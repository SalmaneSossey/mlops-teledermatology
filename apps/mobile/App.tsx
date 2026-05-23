import { Ionicons } from "@expo/vector-icons";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  View
} from "react-native";

import { deleteValue, loadValue, saveValue } from "./src/storage";
import { colors, spacing } from "./src/theme";
import { HistoryScreen } from "./src/screens/HistoryScreen";
import { LoginScreen } from "./src/screens/LoginScreen";
import { NewCaseScreen } from "./src/screens/NewCaseScreen";
import { ProfileScreen } from "./src/screens/ProfileScreen";

type TabKey = "new" | "history" | "profile";

const TOKEN_KEY = "telederm_token";
const ROLE_KEY = "telederm_role";

const TABS: Array<{
  key: TabKey;
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
}> = [
  { key: "new", label: "New Case", icon: "add-circle-outline" },
  { key: "history", label: "History", icon: "time-outline" },
  { key: "profile", label: "Profile", icon: "person-circle-outline" }
];

export default function App() {
  const [token, setToken] = useState<string | null>(null);
  const [loadingSession, setLoadingSession] = useState(true);
  const [activeTab, setActiveTab] = useState<TabKey>("new");
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);

  useEffect(() => {
    async function restoreSession() {
      const [savedToken, savedRole] = await Promise.all([loadValue(TOKEN_KEY), loadValue(ROLE_KEY)]);
      if (savedToken && savedRole === "patient") {
        setToken(savedToken);
      }
      setLoadingSession(false);
    }
    void restoreSession();
  }, []);

  async function handleLogin(nextToken: string, role: string) {
    await Promise.all([saveValue(TOKEN_KEY, nextToken), saveValue(ROLE_KEY, role)]);
    setToken(nextToken);
    setActiveTab("new");
  }

  async function handleLogout() {
    await Promise.all([deleteValue(TOKEN_KEY), deleteValue(ROLE_KEY)]);
    setToken(null);
    setActiveTab("new");
  }

  if (loadingSession) {
    return (
      <SafeAreaView style={styles.loadingScreen}>
        <ActivityIndicator color={colors.primary} />
      </SafeAreaView>
    );
  }

  if (!token) {
    return (
      <>
        <StatusBar barStyle="dark-content" />
        <LoginScreen onLogin={handleLogin} />
      </>
    );
  }

  return (
    <SafeAreaView style={styles.shell}>
      <StatusBar barStyle="dark-content" />
      <View style={styles.body}>
        {activeTab === "new" ? (
          <NewCaseScreen
            onSubmitted={() => setHistoryRefreshKey((current) => current + 1)}
            token={token}
          />
        ) : null}
        {activeTab === "history" ? (
          <HistoryScreen refreshKey={historyRefreshKey} token={token} />
        ) : null}
        {activeTab === "profile" ? <ProfileScreen onLogout={handleLogout} token={token} /> : null}
      </View>
      <View style={styles.tabBar}>
        {TABS.map((tab) => {
          const selected = activeTab === tab.key;
          return (
            <Pressable
              key={tab.key}
              onPress={() => setActiveTab(tab.key)}
              style={styles.tabButton}
            >
              <Ionicons
                color={selected ? colors.primary : colors.inactive}
                name={tab.icon}
                size={24}
              />
              <Text style={[styles.tabLabel, selected && styles.tabLabelSelected]}>
                {tab.label}
              </Text>
            </Pressable>
          );
        })}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  shell: {
    backgroundColor: colors.background,
    flex: 1
  },
  body: {
    flex: 1
  },
  loadingScreen: {
    alignItems: "center",
    backgroundColor: colors.background,
    flex: 1,
    justifyContent: "center"
  },
  tabBar: {
    alignItems: "center",
    backgroundColor: colors.surface,
    borderTopColor: colors.border,
    borderTopWidth: 1,
    bottom: 0,
    flexDirection: "row",
    minHeight: 76,
    paddingBottom: spacing.sm,
    paddingTop: spacing.sm,
    position: "absolute",
    width: "100%"
  },
  tabButton: {
    alignItems: "center",
    flex: 1,
    gap: spacing.xs,
    justifyContent: "center",
    minHeight: 60
  },
  tabLabel: {
    color: colors.inactive,
    fontSize: 12,
    fontWeight: "800"
  },
  tabLabelSelected: {
    color: colors.primary
  }
});
