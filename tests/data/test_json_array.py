"""JSON syntax, streaming, and numeric compatibility of the array reader."""

import io
import json

import pytest

import tinyllava.data.readers.json_array as reader


@pytest.mark.parametrize(
    "source",
    [
        "[{},]",
        "[{} {}]",
        "[{}\u00a0,{}]",
        "[",
        "[{}",
        '[{"text": "unfinished',
        "[{}] {}",
        "[{}] []",
        "[{}] garbage",
        '[{"value": NaN}]',
        '[{"value": Infinity}]',
        '[{"value": 01}]',
        "[/* comment */{}]",
    ],
)
def test_rejects_invalid_json(tmp_path, source):
    path = tmp_path / "invalid.json"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON array"):
        list(reader.iter_json_array(str(path)))


@pytest.mark.parametrize(
    "source", [b"", b" ", b"{}", b'{"item": {}}', b"null", b"\xef[{}]", b"\xc2\xa0[{}]"]
)
def test_rejects_non_array_prefix(tmp_path, source):
    path = tmp_path / "not-array.json"
    path.write_bytes(source)
    assert not reader.is_json_array(path)
    with pytest.raises(ValueError, match="Expected a top-level JSON array"):
        list(reader.iter_json_array(str(path)))


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
def test_bom_long_whitespace_and_unicode(tmp_path, monkeypatch, bom):
    path = tmp_path / "array.json"
    samples = [{"text": '你好 🌍 [,] " \\', "nested": [{"ok": True}, None]}]
    path.write_bytes(
        bom
        + b" \t\r\n" * 20000
        + json.dumps(samples, ensure_ascii=False).encode()
        + b"\n"
    )
    monkeypatch.setattr(reader, "_READ_CHUNK_SIZE", 7)
    assert reader.is_json_array(path)
    assert list(reader.iter_json_array(str(path))) == samples


def test_numeric_types_match_stdlib(tmp_path):
    path = tmp_path / "numbers.json"
    source = '[{"id": 184467440737095516160, "nested": [1.25, -2.5e-3, 1e400]}]'
    path.write_text(source)
    actual = list(reader.iter_json_array(str(path)))
    assert actual == json.loads(source)
    assert type(actual[0]["id"]) is int
    assert all(type(value) is float for value in actual[0]["nested"])


@pytest.mark.parametrize("value", ["null", "2", '"text"', "[]", "true"])
def test_rejects_non_object_rows(tmp_path, value):
    path = tmp_path / "rows.json"
    path.write_text(f"[{{}}, {value}]")
    with pytest.raises(TypeError, match="must contain JSON objects"):
        list(reader.iter_json_array(str(path)))


def test_empty_array(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]")
    assert list(reader.iter_json_array(str(path))) == []
    with pytest.raises(ValueError, match="is empty"):
        reader.read_first_json_array_item(str(path))


class ObservedStream(io.BytesIO):
    def close(self):
        self.position_at_close = self.tell()
        super().close()


def test_first_item_does_not_read_whole_file_and_closes(monkeypatch):
    stream = ObservedStream(b'[{"id": "first"},' + b"{} ," * 100000 + b"{}]")
    monkeypatch.setattr(reader, "open", lambda *args: stream, raising=False)
    assert reader.read_first_json_array_item("large.json") == {"id": "first"}
    assert stream.closed
    assert stream.position_at_close < 100000


def test_invalid_prefix_fails_before_reading_whole_file(monkeypatch):
    stream = ObservedStream(b"[x" + b" " * 400000 + b"]")
    monkeypatch.setattr(reader, "open", lambda *args: stream, raising=False)
    with pytest.raises(ValueError, match="Invalid JSON array"):
        list(reader.iter_json_array("broken.json"))
    assert stream.closed
    assert stream.position_at_close < 100000
