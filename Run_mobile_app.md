# Run Mobile App

This runbook is for testing the Expo patient mobile app from WSL2 on a physical Android phone.
The most reliable professor-demo path is now **USB debugging + `adb reverse`**.

## What Worked

- Backend: Docker Desktop running the FastAPI `api-1` container on port `8000`.
- Phone app: Expo Go SDK 56.
- API access from phone: `adb reverse tcp:8000 tcp:8000`.
- Metro/Expo access from phone: `adb reverse tcp:8081 tcp:8081` and Expo localhost mode.
- Physical phone validation succeeded on June 1, 2026 with image upload, prediction, and history.

Using Expo LAN mode from WSL2 can make the phone spin forever because the QR points to a Windows/WSL address such as `192.168.0.105:8081`. Expo tunnel mode also failed during validation with ngrok tunnel errors, so USB debugging is the safest demo setup.

## Professor Demo Quick Path

Use this sequence in front of the professor.

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

## 2. Connect Android Through USB Debugging

On the phone:

1. Enable Developer options.
2. Enable USB debugging.
3. Connect by USB.
4. Set USB mode to File transfer / MTP if needed.
5. Accept the "Allow USB debugging?" prompt.

From WSL:

```bash
adb devices
```

If it shows a device, continue to step 4. If it shows nothing, use the WSL USB passthrough steps below.

## 3. WSL USB Passthrough If `adb devices` Is Empty

Open Windows PowerShell as Administrator:

```powershell
usbipd list
```

Find the phone. During validation it appeared as:

```text
2-4    2717:ff08  Redmi 12  Not shared
```

Bind and attach it:

```powershell
usbipd bind --busid 2-4
usbipd attach --wsl --busid 2-4
```

Back in WSL, check whether the phone is visible:

```bash
lsusb
adb devices
```

If `lsusb` sees the phone but `adb devices` is empty, fix USB permissions. During validation the phone appeared at `/dev/bus/usb/001/002`:

```bash
sudo chmod a+rw /dev/bus/usb/001/002
adb kill-server
adb start-server
adb devices
```

If the path changes after unplugging, find it with:

```bash
lsusb
ls -l /dev/bus/usb/001/*
```

The correct `adb devices` output should look like:

```text
1cba11c57d7b    device
```

If it says `unauthorized`, unlock the phone and accept the USB debugging prompt.

## 4. Reverse Ports For Backend And Expo

From WSL:

```bash
adb reverse tcp:8081 tcp:8081
adb reverse tcp:8000 tcp:8000
adb reverse --list
```

Expected reverse list:

```text
UsbFfs tcp:8081 tcp:8081
UsbFfs tcp:8000 tcp:8000
```

Be careful not to type an extra character in the port command. This is wrong:

```bash
adb reverse tcp:8000 tcp:8000v
```

## 5. Start Expo Through USB

Open a terminal:

```bash
cd apps/mobile
npm install
EXPO_PUBLIC_TELEDERM_API_URL=http://127.0.0.1:8000 npx expo start --localhost --clear
```

Open Expo Go on the phone. If it does not open automatically, scan the Expo QR or manually open the `exp://127.0.0.1:8081` URL that Expo prints.

## 6. Login And Validate The Flow

Use the seeded patient account:

```text
patient@example.com
patient123
```

Recommended demo flow:

1. Create a new case with notes and metadata.
2. Pick a public/sample lesion image from the gallery.
3. Confirm the image preview appears.
4. Tap Submit and predict.
5. Confirm the app moves through creating consultation, uploading image, and running prediction.
6. Capture the prediction result screen with risk, label, warning, and probability bars.
7. Open History and capture the submitted case marked as predicted.

If the app says `Invalid or expired token`, go to Profile, logout, and login again. If it persists, clear Expo Go app storage and restart the app.

## 7. Doctor/Admin Demo Screens

Start Streamlit from the repository root:

```bash
TELEDERM_API_URL=http://localhost:8000 streamlit run src/app/streamlit_client.py --server.address 0.0.0.0 --server.port 8501
```

Open:

```text
http://localhost:8501
```

Doctor login:

```text
doctor@example.com
doctor123
```

Capture the doctor review page with the submitted case, lesion image preview, latest prediction, probabilities, and review form.

Admin login:

```text
admin@example.com
admin123
```

Capture active model, monitoring, recent predictions, and retraining candidates.

## 8. Optional Fallback: API Tunnel

Use this only if USB debugging is not available.

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

If localtunnel hangs or times out, use `localhost.run`:

```bash
ssh -R 80:localhost:8000 nokey@localhost.run
```

It prints a backend URL like:

```text
https://53bcd78be02ea2.lhr.life
```

Verify:

```bash
curl https://53bcd78be02ea2.lhr.life/docs
```

This URL is only for the backend API. Do not paste it into Expo Go.

## 9. Optional Fallback: Expo Tunnel

Open a second terminal:

```bash
cd apps/mobile
npm install
EXPO_PUBLIC_TELEDERM_API_URL=https://strong-rockets-speak.loca.lt npx expo start --tunnel --clear
```

Replace `https://strong-rockets-speak.loca.lt` with the URL printed by `localtunnel`.

Scan the QR code with Expo Go.

## 10. If It Spins

Stop stale Expo processes and restart one clean tunnel:

```bash
pkill -f "expo start" || true
EXPO_PUBLIC_TELEDERM_API_URL=https://strong-rockets-speak.loca.lt npx expo start --tunnel --clear
```

Fully close Expo Go on the phone, reopen it, and scan the fresh QR.

## 11. If Login Cannot Connect

Confirm the API tunnel still works:

```bash
curl https://strong-rockets-speak.loca.lt/docs
```

If it fails, stop and restart the `localtunnel` command. Copy the new URL into the Expo command.

## 12. Notes From Debugging

- Expo Go must match the project SDK. This app currently uses Expo SDK 56.
- USB debugging with `adb reverse` is the preferred physical-phone demo route.
- `localtunnel` can print a URL and still time out from the phone or `curl`.
- Expo tunnel can fail with ngrok messages such as `failed to start tunnel`, `session closed`, or `remote gone away`.
- `localhost.run` can expose the backend, but it does not open the Expo app.
- Safe-area handling uses `react-native-safe-area-context`; the deprecated React Native `SafeAreaView` warning should no longer appear.
- Image upload uses `expo-file-system/legacy` multipart upload because newer React Native rejected the old FormData object shape.
- Expo image picker may warn that `ImagePicker.MediaTypeOptions` is deprecated. This warning does not block the demo.

## 13. Physical Phone Validation Checklist

- Login succeeds with the seeded patient account.
- Gallery upload shows an image preview before submission.
- Submit shows progress through consultation creation, image upload, and model prediction.
- Prediction result shows the risk badge, predicted label, warning, and probability bars.
- The new case appears in patient history after submission.
- Doctor review shows the lesion image preview and probability table/bar chart in the Streamlit UI.
