import { fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client.js";
import { renderWithProviders } from "../tests/testUtils.jsx";
import UploadPage from "./UploadPage.jsx";

const { uploadChequeMock } = vi.hoisted(() => ({ uploadChequeMock: vi.fn() }));
vi.mock("../api/cheques.js", () => ({ uploadCheque: uploadChequeMock }));

function makeFile(name, type, size = 1024) {
  const file = new File(["x".repeat(size)], name, { type });
  return file;
}

describe("UploadPage", () => {
  it("rejects an unsupported file type before ever calling the API", () => {
    renderWithProviders(<UploadPage />);
    const input = document.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [makeFile("cheque.gif", "image/gif")] } });

    expect(screen.getByTestId("upload-validation-error")).toHaveTextContent("Only JPEG, PNG, TIFF and PDF");
    expect(uploadChequeMock).not.toHaveBeenCalled();
  });

  it("uploads successfully and shows the returned Processing ID", async () => {
    uploadChequeMock.mockResolvedValueOnce({
      success: true, cheque_id: "CHK-2026-000099", status: "UPLOADED", message: "Cheque uploaded successfully.",
    });
    renderWithProviders(<UploadPage />);
    const input = document.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [makeFile("cheque.png", "image/png")] } });

    fireEvent.click(screen.getByRole("button", { name: /^upload$/i }));

    await waitFor(() => expect(screen.getByTestId("upload-success")).toBeInTheDocument());
    expect(screen.getByText("CHK-2026-000099")).toBeInTheDocument();
  });

  it("shows a clear error message when the backend rejects the upload", async () => {
    uploadChequeMock.mockRejectedValueOnce(new ApiError("Only JPEG, PNG and PDF files are supported.", { status: 415, code: "INVALID_FILE_TYPE" }));
    renderWithProviders(<UploadPage />);
    const input = document.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [makeFile("cheque.png", "image/png")] } });
    fireEvent.click(screen.getByRole("button", { name: /^upload$/i }));

    await waitFor(() => expect(screen.getByTestId("upload-api-error")).toBeInTheDocument());
  });
});
