# Tauri + React

This template should help get you started developing with Tauri and React in Vite.

## Platform Build & Setup Guides

### macOS / Linux / Windows

- **Build & Run (Dev):**
  ```sh
  npm run tauri dev
  ```
- **Build (Release):**
  ```sh
  npm run tauri build
  ```

### Android

- **Setup Android:**
  ```sh
  npx tauri android dev --open
  ```
- **Build APK:**
  ```sh
  npx tauri android build
  ```
- **Native Plugins:**
  ```sh
  tauri plugin android init
  ```

### iOS (macOS only)

- **Setup iOS:**
  ```sh
  npx tauri ios dev --open
  ```
- **Build IPA:**
  ```sh
  npx tauri ios build
  ```
- **Native Plugins:**
  ```sh
  tauri plugin ios init
  ```

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
