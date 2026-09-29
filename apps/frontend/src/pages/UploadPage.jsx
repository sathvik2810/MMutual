import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadCheque } from "../api/cheques.js";
import { useNotifications } from "../layouts/NotificationContext.jsx";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/tiff", "application/pdf"];
const MAX_SIZE_MB = 10;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);
  const navigate = useNavigate();
  const { notify } = useNotifications();

  const validateAndSetFile = useCallback((selected) => {
    setResult(null);
    setUploadError(null);
    if (!selected) {
      setFile(null);
      return;
    }
    if (!ACCEPTED_TYPES.includes(selected.type)) {
      setValidationError("Only JPEG, PNG, TIFF and PDF files are supported.");
      setFile(null);
      return;
    }
    if (selected.size > MAX_SIZE_MB * 1024 * 1024) {
      setValidationError(`File exceeds the ${MAX_SIZE_MB} MB upload limit.`);
      setFile(null);
      return;
    }
    setValidationError(null);
    setFile(selected);
  }, []);

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    validateAndSetFile(event.dataTransfer.files?.[0]);
  }

  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      const response = await uploadCheque(file);
      setResult(response);
      notify(`Cheque ${response.cheque_id} uploaded successfully.`, "success");
    } catch (err) {
      setUploadError(err);
      notify(err.message || "Upload failed.", "error");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Upload Cheque</h1>
        <p className="text-sm text-slate-500">Upload a cheque image or PDF for automated processing.</p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-10 text-center transition ${
          dragActive ? "border-slate-500 bg-slate-100" : "border-slate-300 bg-white"
        }`}
      >
        <p className="text-sm text-slate-600">Drag and drop a cheque image or PDF here</p>
        <p className="mt-1 text-xs text-slate-400">JPEG, PNG, TIFF or PDF, up to {MAX_SIZE_MB} MB</p>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="mt-4 rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
        >
          Choose file
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.tif,.tiff,.pdf"
          className="hidden"
          onChange={(e) => validateAndSetFile(e.target.files?.[0])}
        />
      </div>

      {validationError && (
        <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700" data-testid="upload-validation-error">
          {validationError}
        </p>
      )}

      {file && (
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <p className="text-sm font-medium text-slate-900">{file.name}</p>
          <p className="text-xs text-slate-500">
            {formatBytes(file.size)} · {file.type || "unknown type"}
          </p>
          <button
            type="button"
            onClick={handleUpload}
            disabled={uploading}
            className="mt-3 rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {uploading ? "Uploading…" : "Upload"}
          </button>
        </div>
      )}

      {uploadError && (
        <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700" data-testid="upload-api-error">
          {uploadError.message}
        </p>
      )}

      {result && (
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4" data-testid="upload-success">
          <p className="text-sm font-medium text-emerald-800">{result.message}</p>
          <p className="mt-1 text-sm text-emerald-700">
            Processing ID: <span className="font-mono">{result.cheque_id}</span>
          </p>
          <button
            type="button"
            onClick={() => navigate(`/cheques/${result.cheque_id}`)}
            className="mt-3 rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-700"
          >
            Continue to processing →
          </button>
        </div>
      )}
    </div>
  );
}
