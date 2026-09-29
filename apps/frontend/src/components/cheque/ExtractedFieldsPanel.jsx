import { Card } from "../common/Card.jsx";
import { EmptyState } from "../common/States.jsx";
import { formatPercent, titleCase } from "../../utils/format.js";

const DISPLAY_FIELDS = ["cheque_number", "date", "payee_name", "amount", "amount_in_words", "account_number", "ifsc", "micr"];

export function ExtractedFieldsPanel({ ocr, extraction }) {
  if (!ocr) {
    return (
      <Card title="OCR / Extraction">
        <EmptyState title="Not run yet" message="OCR has not been run for this cheque." />
      </Card>
    );
  }

  const fields = extraction?.fields ?? {};
  const fieldNames = [...new Set([...DISPLAY_FIELDS, ...Object.keys(fields)])];
  const consistency = extraction?.validation_results?.amount_consistency;

  return (
    <Card
      title="OCR / Extraction"
      subtitle={`Engine ${ocr.engine_name ?? "Unavailable"} ${ocr.engine_version ?? ""} | OCR confidence ${formatPercent(ocr.average_confidence)}`}
    >
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div>
          <p className="text-xs text-slate-500">OCR Status</p>
          <p className="text-sm font-medium text-slate-900">{titleCase(ocr.ocr_status)}</p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Extraction Status</p>
          <p className="text-sm font-medium text-slate-900">{titleCase(extraction?.extraction_status)}</p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Template</p>
          <p className="text-sm font-medium text-slate-900">{extraction?.template ?? "—"}</p>
        </div>
      </div>

      {extraction?.warnings?.length > 0 && (
        <div className="mt-3 rounded-md bg-amber-50 p-3 text-sm text-amber-800" aria-label="Extraction warnings">
          <p className="font-medium">Extraction warnings</p>
          <ul className="mt-1 list-disc pl-5">{extraction.warnings.map((warning, i) => <li key={i}>{warning}</li>)}</ul>
        </div>
      )}
      {consistency && (
        <p className="mt-3 text-sm text-slate-700" data-testid="amount-consistency">
          Amount consistency: <strong>{titleCase(consistency.status)}</strong>
          {consistency.numeric_amount != null && consistency.words_amount != null &&
            ` (numeric ${consistency.numeric_amount}; words ${consistency.words_amount})`}
        </p>
      )}

      {(extraction?.missing_fields?.length > 0 || extraction?.ambiguous_fields?.length > 0) && (
        <div className="mt-3 space-y-1 text-xs">
          {extraction.missing_fields.length > 0 && (
            <p className="text-red-600">Missing fields: {extraction.missing_fields.join(", ")}</p>
          )}
          {extraction.ambiguous_fields.length > 0 && (
            <p className="text-amber-600">Ambiguous fields: {extraction.ambiguous_fields.join(", ")}</p>
          )}
        </div>
      )}

      {fieldNames.length > 0 ? (
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead>
              <tr className="text-left text-xs uppercase text-slate-500">
                <th className="py-2 pr-4">Field</th>
                <th className="py-2 pr-4">Normalized Value</th>
                <th className="py-2 pr-4">Raw (OCR) Value</th>
                <th className="py-2 pr-4">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {fieldNames.map((name) => {
                const f = fields[name] ?? {};
                const uncertain = f.value != null && (f.reliability === "UNCERTAIN" || f.confidence == null || f.confidence < 40);
                return (
                  <tr key={name}>
                    <td className="py-2 pr-4 font-medium text-slate-700">{titleCase(name)}</td>
                    <td className="py-2 pr-4 text-slate-900">{f.value == null || uncertain ? <span className="text-amber-700">Not reliably detected</span> : f.value}</td>
                    <td className="py-2 pr-4 text-slate-500">{f.raw_value ?? "—"}</td>
                    <td className="py-2 pr-4 text-slate-500">{formatPercent(f.confidence)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="mt-4 text-sm text-slate-500">No field-level extraction data available.</p>
      )}
      {ocr.raw_text && (
        <details className="mt-4 rounded-md border border-slate-200 p-3">
          <summary className="cursor-pointer text-sm font-medium text-slate-700">Raw OCR text</summary>
          <pre className="mt-2 whitespace-pre-wrap text-xs text-slate-600">{ocr.raw_text}</pre>
        </details>
      )}
    </Card>
  );
}
