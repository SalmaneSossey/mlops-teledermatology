# Run Mobile App

This runbook is for testing the Expo patient mobile app from WSL2 on a physical Android phone.

## What Worked

- Backend: Docker Desktop running the FastAPI `api-1` container on port `8000`.
- Phone app: Expo Go SDK 56.
- API access from phone: `localtunnel` to expose FastAPI.
- Metro/Expo access from phone: Expo tunnel mode, not LAN mode.

Using Expo LAN mode from WSL2 can make the phone spin forever because the QR points to a Windows/WSL address such as `192.168.0.105:8081`. Tunnel mode avoids that.

## 1. Start Backend

From the repository root:

```bash
docker compose up -d api postgres
```

Check that FastAPI is healthy locally:

```bash
curl http://localhost:8000/docs
```

Docker Desktop should show `api-1` mapped as `8000:8000`.

## 2. Start API Tunnel

Open a terminal in the repository root:

```bash
npx --yes localtunnel --port 8000
```

Keep this terminal open. It prints a URL like:

```text
https://strong-rockets-speak.loca.lt
```

Verify it:

```bash
curl https://strong-rockets-speak.loca.lt/docs
```

Use the URL that `localtunnel` prints in your own session. Do not add `https://` twice.

Correct:

```text
https://strong-rockets-speak.loca.lt
```

Wrong:

```text
https://https://strong-rockets-speak.loca.lt
```

## 3. Start Expo

Open a second terminal:

```bash
cd apps/mobile
npm install
EXPO_PUBLIC_TELEDERM_API_URL=https://strong-rockets-speak.loca.lt npx expo start --tunnel --clear
```

Replace `https://strong-rockets-speak.loca.lt` with the URL printed by `localtunnel`.

Scan the QR code with Expo Go.

## 4. Login

Use the seeded patient account:

```text
patient@example.com
patient123
```

## 5. If It Spins

Stop stale Expo processes and restart one clean tunnel:

```bash
pkill -f "expo start" || true
EXPO_PUBLIC_TELEDERM_API_URL=https://strong-rockets-speak.loca.lt npx expo start --tunnel --clear
```

Fully close Expo Go on the phone, reopen it, and scan the fresh QR.

## 6. If Login Cannot Connect

Confirm the API tunnel still works:

```bash
curl https://strong-rockets-speak.loca.lt/docs
```

If it fails, stop and restart the `localtunnel` command. Copy the new URL into the Expo command.

## 7. Notes From Debugging

- Expo Go must match the project SDK. This app currently uses Expo SDK 56.
- Use Expo tunnel mode for the phone demo.
- Use `localtunnel` for the FastAPI backend from WSL2.
- Keep the `localtunnel` terminal open while using the app.
- Safe-area handling uses `react-native-safe-area-context`; the deprecated React Native `SafeAreaView` warning should no longer appear.
- Image upload uses `expo-file-system/legacy` multipart upload because newer React Native rejected the old FormData object shape.

## 8. Physical Phone Validation Checklist

- Login succeeds with the seeded patient account.
- Gallery upload shows an image preview before submission.
- Submit shows progress through consultation creation, image upload, and model prediction.
- Prediction result shows the risk badge, predicted label, warning, and probability bars.
- The new case appears in patient history after submission.
- Doctor review shows the lesion image preview and probability table/bar chart in the Streamlit UI.
