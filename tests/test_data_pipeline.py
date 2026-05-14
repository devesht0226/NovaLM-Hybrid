from pathlib import Path

from src.data_pipeline.build_index import build_chunks
from src.data_pipeline.clean import clean_text_lines
from src.data_pipeline.dedup import deduplicate
from src.data_pipeline.ingest import load_raw_texts
from src.data_pipeline.split import split_lines
from src.utils.chunks_json import load_chunks_json


def test_build_chunks_overlap():
    lines = ["a", "b", "c", "d", "e"]
    no_ov = build_chunks(lines, chunk_size=2, overlap_lines=0)
    assert no_ov == ["a b", "c d", "e"]
    ov = build_chunks(lines, chunk_size=2, overlap_lines=1)
    assert ov[0] == "a b"
    assert "b c" in ov


def test_load_chunks_json_wrapped(tmp_path: Path):
    p = tmp_path / "chunks.json"
    p.write_text('{"chunks": [{"text": " hello "}, "world"]}', encoding="utf-8")
    assert load_chunks_json(p) == ["hello", "world"]


def test_clean_and_dedup_and_split():
    txt = " Hello   world \n\nHELLO world\nx\n"
    lines = clean_text_lines(txt)
    assert "hello world" in lines
    uniq = deduplicate(lines)
    assert uniq.count("hello world") == 1
    train, val, test = split_lines(uniq, seed=1)
    assert len(train) + len(val) + len(test) == len(uniq)


def test_load_raw_texts_recursive(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.txt").write_text("one", encoding="utf-8")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "two.txt").write_text("two", encoding="utf-8")
    texts = load_raw_texts(str(tmp_path))
    assert len(texts) == 2
    assert "one" in texts
    assert "two" in texts
