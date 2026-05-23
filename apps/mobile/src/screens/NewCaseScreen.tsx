import { useState } from "react";
import {
  ActivityIndicator,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View
} from "react-native";
import * as ImagePicker from "expo-image-picker";

import {
  createConsultation,
  predictConsultation,
  uploadConsultationImage
} from "../api";
import { ProbabilityBars } from "../components/ProbabilityBars";
import { Badge } from "../components/RiskBadge";
import { SectionHeader } from "../components/SectionHeader";
import { colors, radii, spacing } from "../theme";
import { ClinicalMetadata, PredictionResponse, SelectedImage } from "../types";

type NewCaseScreenProps = {
  token: string;
  onSubmitted: () => void;
};

type MetadataForm = {
  age: string;
  region: string;
  gender: string;
  fitspatrick: string;
  diameter_1: string;
  diameter_2: string;
  itch: string;
  grew: string;
  hurt: string;
  changed: string;
  bleed: string;
  elevation: string;
};

const REGIONS = ["FACE", "ARM", "BACK", "CHEST", "LEG", "HAND"];
const BOOLEAN_OPTIONS = ["False", "True", "UNK"];
const FITZPATRICK = ["1", "2", "3", "4", "5", "6"];

const DEFAULT_FORM: MetadataForm = {
  age: "55",
  region: "FACE",
  gender: "",
  fitspatrick: "3",
  diameter_1: "6",
  diameter_2: "5",
  itch: "False",
  grew: "True",
  hurt: "False",
  changed: "True",
  bleed: "False",
  elevation: "True"
};

function numericValue(value: string, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function toMetadata(form: MetadataForm): ClinicalMetadata {
  return {
    age: numericValue(form.age, 0),
    region: form.region,
    itch: form.itch,
    grew: form.grew,
    hurt: form.hurt,
    changed: form.changed,
    bleed: form.bleed,
    elevation: form.elevation,
    gender: form.gender || null,
    fitspatrick: numericValue(form.fitspatrick, 0),
    diameter_1: numericValue(form.diameter_1, 0),
    diameter_2: numericValue(form.diameter_2, 0),
    skin_cancer_history: "UNK",
    cancer_history: "UNK",
    smoke: "UNK",
    drink: "UNK",
    pesticide: "UNK"
  };
}

type OptionRowProps = {
  label: string;
  options: string[];
  value: string;
  onChange: (value: string) => void;
};

function OptionRow({ label, options, value, onChange }: OptionRowProps) {
  return (
    <View style={styles.field}>
      <Text style={styles.label}>{label}</Text>
      <View style={styles.optionRow}>
        {options.map((option) => {
          const selected = option === value;
          return (
            <Pressable
              key={option}
              onPress={() => onChange(option)}
              style={[styles.option, selected && styles.optionSelected]}
            >
              <Text style={[styles.optionText, selected && styles.optionTextSelected]}>
                {option || "UNK"}
              </Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

export function NewCaseScreen({ token, onSubmitted }: NewCaseScreenProps) {
  const [form, setForm] = useState<MetadataForm>(DEFAULT_FORM);
  const [notes, setNotes] = useState("");
  const [image, setImage] = useState<SelectedImage | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function updateForm(key: keyof MetadataForm, value: string) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function chooseImage(source: "camera" | "gallery") {
    setError(null);
    const permission =
      source === "camera"
        ? await ImagePicker.requestCameraPermissionsAsync()
        : await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (permission.status !== "granted") {
      setError("Image permission was not granted.");
      return;
    }
    const result =
      source === "camera"
        ? await ImagePicker.launchCameraAsync({ allowsEditing: true, quality: 0.9 })
        : await ImagePicker.launchImageLibraryAsync({
            allowsEditing: true,
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            quality: 0.9
          });
    if (result.canceled || result.assets.length === 0) {
      return;
    }
    const asset = result.assets[0];
    setImage({
      uri: asset.uri,
      fileName: asset.fileName ?? `lesion-${Date.now()}.jpg`,
      mimeType: asset.mimeType ?? "image/jpeg"
    });
  }

  async function submitCase() {
    if (!image) {
      setError("Add a lesion image before submitting.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const consultation = await createConsultation(token, notes, toMetadata(form));
      await uploadConsultationImage(token, consultation.id, image);
      const result = await predictConsultation(token, consultation.id);
      setPrediction(result);
      onSubmitted();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Submission failed");
    } finally {
      setSubmitting(false);
    }
  }

  function resetForm() {
    setPrediction(null);
    setNotes("");
    setImage(null);
    setForm(DEFAULT_FORM);
    setError(null);
  }

  return (
    <ScrollView contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Text style={styles.title}>New Case</Text>
        <Text style={styles.subtitle}>Patient lesion assessment</Text>
      </View>

      {prediction ? (
        <View style={styles.resultCard}>
          <View style={styles.resultHeader}>
            <Badge label={prediction.risk_level} tone="risk" />
            <Text style={styles.predictedLabel}>{prediction.predicted_label}</Text>
          </View>
          <Text style={styles.warning}>{prediction.warning}</Text>
          <SectionHeader title="Probabilities" />
          <ProbabilityBars probabilities={prediction.probabilities} />
          <Pressable onPress={resetForm} style={styles.secondaryButton}>
            <Text style={styles.secondaryButtonText}>Start another case</Text>
          </Pressable>
        </View>
      ) : null}

      {!prediction ? (
        <>
          <View style={styles.formBlock}>
            <SectionHeader title="Clinical Notes" />
            <TextInput
              multiline
              onChangeText={setNotes}
              placeholder="Symptoms, duration, concerns"
              placeholderTextColor={colors.inactive}
              style={[styles.input, styles.notes]}
              value={notes}
            />

            <SectionHeader title="Image" />
            <View style={styles.imageActions}>
              <Pressable onPress={() => chooseImage("gallery")} style={styles.imageButton}>
                <Text style={styles.imageButtonText}>Gallery</Text>
              </Pressable>
              <Pressable onPress={() => chooseImage("camera")} style={styles.imageButton}>
                <Text style={styles.imageButtonText}>Camera</Text>
              </Pressable>
            </View>
            {image ? <Image source={{ uri: image.uri }} style={styles.preview} /> : null}

            <SectionHeader title="Metadata" />
            <View style={styles.twoColumn}>
              <View style={styles.field}>
                <Text style={styles.label}>Age</Text>
                <TextInput
                  keyboardType="numeric"
                  onChangeText={(value) => updateForm("age", value)}
                  style={styles.input}
                  value={form.age}
                />
              </View>
              <View style={styles.field}>
                <Text style={styles.label}>Diameter 1</Text>
                <TextInput
                  keyboardType="numeric"
                  onChangeText={(value) => updateForm("diameter_1", value)}
                  style={styles.input}
                  value={form.diameter_1}
                />
              </View>
              <View style={styles.field}>
                <Text style={styles.label}>Diameter 2</Text>
                <TextInput
                  keyboardType="numeric"
                  onChangeText={(value) => updateForm("diameter_2", value)}
                  style={styles.input}
                  value={form.diameter_2}
                />
              </View>
            </View>
            <OptionRow
              label="Region"
              onChange={(value) => updateForm("region", value)}
              options={REGIONS}
              value={form.region}
            />
            <OptionRow
              label="Gender"
              onChange={(value) => updateForm("gender", value)}
              options={["", "FEMALE", "MALE"]}
              value={form.gender}
            />
            <OptionRow
              label="Fitzpatrick"
              onChange={(value) => updateForm("fitspatrick", value)}
              options={FITZPATRICK}
              value={form.fitspatrick}
            />
            <OptionRow
              label="Itch"
              onChange={(value) => updateForm("itch", value)}
              options={BOOLEAN_OPTIONS}
              value={form.itch}
            />
            <OptionRow
              label="Grew"
              onChange={(value) => updateForm("grew", value)}
              options={BOOLEAN_OPTIONS}
              value={form.grew}
            />
            <OptionRow
              label="Hurt"
              onChange={(value) => updateForm("hurt", value)}
              options={BOOLEAN_OPTIONS}
              value={form.hurt}
            />
            <OptionRow
              label="Changed"
              onChange={(value) => updateForm("changed", value)}
              options={BOOLEAN_OPTIONS}
              value={form.changed}
            />
            <OptionRow
              label="Bleed"
              onChange={(value) => updateForm("bleed", value)}
              options={BOOLEAN_OPTIONS}
              value={form.bleed}
            />
            <OptionRow
              label="Elevation"
              onChange={(value) => updateForm("elevation", value)}
              options={BOOLEAN_OPTIONS}
              value={form.elevation}
            />
          </View>
          {error ? <Text style={styles.error}>{error}</Text> : null}
          <Pressable disabled={submitting} onPress={submitCase} style={styles.primaryButton}>
            {submitting ? (
              <ActivityIndicator color={colors.surface} />
            ) : (
              <Text style={styles.primaryButtonText}>Submit and predict</Text>
            )}
          </Pressable>
        </>
      ) : null}
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
  formBlock: {
    gap: spacing.md
  },
  field: {
    gap: spacing.sm
  },
  label: {
    color: colors.text,
    fontSize: 14,
    fontWeight: "700"
  },
  input: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.sm,
    borderWidth: 1,
    color: colors.text,
    fontSize: 16,
    minHeight: 44,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm
  },
  notes: {
    minHeight: 96,
    textAlignVertical: "top"
  },
  imageActions: {
    flexDirection: "row",
    gap: spacing.md
  },
  imageButton: {
    alignItems: "center",
    backgroundColor: colors.surfaceMuted,
    borderColor: colors.border,
    borderRadius: radii.sm,
    borderWidth: 1,
    flex: 1,
    minHeight: 44,
    justifyContent: "center"
  },
  imageButtonText: {
    color: colors.primaryDark,
    fontSize: 15,
    fontWeight: "800"
  },
  preview: {
    aspectRatio: 1.4,
    borderRadius: radii.md,
    width: "100%"
  },
  twoColumn: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.md
  },
  optionRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.sm
  },
  option: {
    borderColor: colors.border,
    borderRadius: radii.sm,
    borderWidth: 1,
    minHeight: 36,
    minWidth: 66,
    paddingHorizontal: spacing.md,
    justifyContent: "center"
  },
  optionSelected: {
    backgroundColor: colors.primary,
    borderColor: colors.primary
  },
  optionText: {
    color: colors.text,
    fontSize: 13,
    fontWeight: "700",
    textAlign: "center"
  },
  optionTextSelected: {
    color: colors.surface
  },
  primaryButton: {
    alignItems: "center",
    backgroundColor: colors.primary,
    borderRadius: radii.sm,
    justifyContent: "center",
    minHeight: 52
  },
  primaryButtonText: {
    color: colors.surface,
    fontSize: 16,
    fontWeight: "900"
  },
  secondaryButton: {
    alignItems: "center",
    borderColor: colors.primary,
    borderRadius: radii.sm,
    borderWidth: 1,
    justifyContent: "center",
    minHeight: 46
  },
  secondaryButtonText: {
    color: colors.primaryDark,
    fontSize: 15,
    fontWeight: "800"
  },
  resultCard: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.md,
    borderWidth: 1,
    gap: spacing.lg,
    padding: spacing.lg
  },
  resultHeader: {
    gap: spacing.md
  },
  predictedLabel: {
    color: colors.text,
    fontSize: 36,
    fontWeight: "900",
    letterSpacing: 0
  },
  warning: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 20
  },
  error: {
    color: colors.danger,
    fontSize: 14,
    fontWeight: "700"
  }
});
