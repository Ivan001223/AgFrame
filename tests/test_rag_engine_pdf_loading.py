from __future__ import annotations

from app.skills.rag import rag_engine


def _bare_engine() -> rag_engine.RAGEngine:
    return object.__new__(rag_engine.RAGEngine)


def test_pdf_loading_uses_pypdf_before_any_ocr_path(monkeypatch):
    engine = _bare_engine()

    monkeypatch.setattr(rag_engine, "_tesseract_available", lambda: True)
    monkeypatch.setattr(rag_engine, "_poppler_available", lambda: True)
    monkeypatch.setattr(
        rag_engine,
        "_extract_pdf_text_with_tesseract",
        lambda path: (_ for _ in ()).throw(AssertionError("tesseract OCR should be skipped when pypdf succeeds")),
    )
    monkeypatch.setattr(rag_engine, "_extract_pdf_text_with_pypdf", lambda path: "plain pdf text")
    monkeypatch.setattr(
        rag_engine,
        "_get_ocr_engine",
        lambda: (_ for _ in ()).throw(AssertionError("OCR should not be used")),
    )

    docs = engine.load_documents("/tmp/report.pdf")

    assert len(docs) == 1
    assert docs[0].page_content == "plain pdf text"


def test_pdf_loading_falls_back_to_ocr_without_tesseract(monkeypatch):
    engine = _bare_engine()
    calls: list[str] = []

    class _OCR:
        def process_file(self, file_path: str) -> str:
            calls.append(file_path)
            return "ocr text"

    monkeypatch.setattr(rag_engine, "_tesseract_available", lambda: False)
    monkeypatch.setattr(rag_engine, "_poppler_available", lambda: False)
    monkeypatch.setattr(
        rag_engine,
        "_extract_pdf_text_with_tesseract",
        lambda path: (_ for _ in ()).throw(AssertionError("tesseract OCR should be skipped without tesseract")),
    )
    monkeypatch.setattr(rag_engine, "_extract_pdf_text_with_pypdf", lambda path: "")
    monkeypatch.setattr(rag_engine, "_get_ocr_engine", lambda: _OCR())

    docs = engine.load_documents("/tmp/scan.pdf")

    assert len(docs) == 1
    assert docs[0].page_content == "ocr text"
    assert calls == ["/tmp/scan.pdf"]


def test_pdf_loading_uses_tesseract_ocr_when_local_prereqs_exist(monkeypatch):
    engine = _bare_engine()

    monkeypatch.setattr(rag_engine, "_tesseract_available", lambda: True)
    monkeypatch.setattr(rag_engine, "_poppler_available", lambda: True)
    monkeypatch.setattr(rag_engine, "_extract_pdf_text_with_pypdf", lambda path: "")
    monkeypatch.setattr(
        rag_engine,
        "_extract_pdf_text_with_tesseract",
        lambda path: "ocr only text",
    )
    monkeypatch.setattr(
        rag_engine,
        "_get_ocr_engine",
        lambda: (_ for _ in ()).throw(AssertionError("local OCR should not be used when tesseract OCR succeeds")),
    )

    docs = engine.load_documents("/tmp/table.pdf")

    assert len(docs) == 1
    assert docs[0].page_content == "ocr only text"


def test_image_loading_uses_tesseract_before_local_ocr(monkeypatch):
    engine = _bare_engine()

    monkeypatch.setattr(rag_engine, "_tesseract_available", lambda: True)
    monkeypatch.setattr(rag_engine, "_extract_image_text_with_tesseract", lambda path: "image text")
    monkeypatch.setattr(
        rag_engine,
        "_get_ocr_engine",
        lambda: (_ for _ in ()).throw(AssertionError("local OCR should not be used when tesseract succeeds")),
    )

    docs = engine.load_documents("/tmp/photo.png")

    assert len(docs) == 1
    assert docs[0].page_content == "image text"
