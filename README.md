# ReelClaw

ReelClaw is an iOS app + backend that lets a user:

1) Select clips from their phone
2) Provide a reference reel/video link (Instagram, YouTube, etc.) or upload a reference video
3) Generate multiple edited variants that replicate the reference pacing/structure
4) Get notified when the job finishes

This repository is organized into two main folders:

- `ios/` – Xcode project + iOS client
- `backend/` – FastAPI API + AWS Batch worker + reel replication pipeline (no AI image generation)

## Backend (local dev)

See `backend/README.md`.

## iOS (local dev)

Open `ios/ReelClaw/ReelClaw.xcodeproj` in Xcode.
