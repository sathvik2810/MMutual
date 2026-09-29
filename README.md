# AI-Powered Cheque Scanning, Validation & Fraud Detection System

Mass Mutual Project

This repository contains the documentation, architecture, source code, data, machine learning components, testing resources, and deployment configuration for an AI-powered cheque processing and fraud detection system.

Detailed project documentation is maintained under the `docs/` directory.

## Prototype OCR pipeline

The upload pipeline uses PaddleOCR first and retries with Tesseract when PaddleOCR is unavailable or produces empty, partial, or low confidence output. Use Python 3.13 (64-bit) for the pinned CPU PaddlePaddle/PaddleOCR release in `apps/backend/requirements.txt`; PaddlePaddle currently provides Windows wheels through Python 3.13. PaddleOCR downloads its English recognition models on the first OCR request. Tesseract remains a native system dependency; on Windows install it with `winget install UB-Mannheim.TesseractOCR` and set `tesseract_cmd_path` in `apps/backend/app/core/config.py` if it is installed elsewhere.

Run the backend from the repository root:

```powershell
py -3.13 -m venv apps/backend/.venv
apps/backend/.venv/Scripts/python.exe -m pip install -r apps/backend/requirements.txt
apps/backend/.venv/Scripts/python.exe -m uvicorn app.main:app --reload --app-dir apps/backend --port 8000
```

Run the frontend in another terminal:

```powershell
cd apps/frontend
npm.cmd ci
npm.cmd run dev
```

Run the complete backend and frontend test suites:

```powershell
cd apps/backend
.venv\Scripts\python.exe -m pytest -v
cd ../frontend
npm.cmd test
```

Project documentation (requirements, architecture, and design specifications) is complete under `docs/`. Implementation is proceeding milestone by milestone; see `docs/35_MVP_Roadmap.md` (Section 4.1) and `docs/36_Development_Guidelines.md` for the development approach. All code, configuration, and data in this repository must continue to follow the constraints documented there — most importantly, no real customer or banking information may be used; only synthetic/mock data is permitted.
