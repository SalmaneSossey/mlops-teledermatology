# Telederm Patient Mobile

Expo TypeScript patient app for the FastAPI teledermatology demo.

## Local Demo

Start the backend from the repository root:

```bash
docker compose up api postgres
```

Find your computer LAN IP, then start Expo:

```bash
cd apps/mobile
npm install
EXPO_PUBLIC_TELEDERM_API_URL=http://<computer-lan-ip>:8000 npm start
```

Use the seeded demo account:

```text
patient@example.com / patient123
```

For Android emulator-only testing, `http://10.0.2.2:8000` can usually reach the host machine.
