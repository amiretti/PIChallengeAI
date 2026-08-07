from app.domain.chunking import split_into_chunks


def test_creates_one_chunk_per_paragraph():
    text = "First story.\nSecond story.\nThird story."

    chunks = split_into_chunks(text)

    assert len(chunks) == 3
    assert [chunk.text for chunk in chunks] == [
        "First story.",
        "Second story.",
        "Third story.",
    ]


def test_ignores_blank_lines_and_surrounding_whitespace():
    text = "\n  First story.  \n\n\n   \nSecond story.\n\n"

    chunks = split_into_chunks(text)

    assert [chunk.text for chunk in chunks] == ["First story.", "Second story."]


def test_assigns_sequential_deterministic_ids():
    text = "One.\nTwo."

    assert [chunk.id for chunk in split_into_chunks(text)] == ["chunk-0", "chunk-1"]


def test_is_idempotent_across_calls():
    text = "One.\nTwo."

    assert split_into_chunks(text) == split_into_chunks(text)


def test_returns_empty_list_for_blank_text():
    assert split_into_chunks("   \n\n  ") == []
