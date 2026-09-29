import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ExtractedFieldsPanel } from "./ExtractedFieldsPanel.jsx";

describe("ExtractedFieldsPanel", () => {
  it("renders backend OCR evidence, field confidence, warnings, and amount validation separately", () => {
    render(<ExtractedFieldsPanel
      ocr={{ engine_name: "PaddleOCR", engine_version: "3.7.0", average_confidence: 88, ocr_status: "COMPLETED", raw_text: "Amount: 1500" }}
      extraction={{
        extraction_status: "PARTIAL",
        warnings: ["micr was not detected reliably."],
        validation_results: { amount_consistency: { status: "FAIL", numeric_amount: 1500, words_amount: 1000 } },
        fields: {
          amount: { value: 1500, raw_value: "Rs 1,500", confidence: 91, reliability: "RELIABLE" },
          payee_name: { value: "Wrong?", raw_value: "Wr0ng?", confidence: 18, reliability: "UNCERTAIN" },
        },
      }}
    />);

    expect(screen.getByText(/Engine PaddleOCR/)).toBeInTheDocument();
    expect(screen.getByText(/OCR confidence 88/)).toBeInTheDocument();
    expect(screen.getByText("1500")).toBeInTheDocument();
    expect(screen.getAllByText("Not reliably detected").length).toBeGreaterThan(0);
    expect(screen.getByText("micr was not detected reliably.")).toBeInTheDocument();
    expect(screen.getByTestId("amount-consistency")).toHaveTextContent("Fail");
    expect(screen.getByText("Amount: 1500")).toBeInTheDocument();
  });
});
