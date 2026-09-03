from tinyllava.serve import app


def test_add_turn_uses_hf_messages(monkeypatch):
    captured = {}

    def fake_generate_response(**kwargs):
        captured.update(kwargs)
        return "A cat."

    monkeypatch.setattr(app, "generate_response", fake_generate_response)

    messages, history = app.add_turn(
        model=object(),
        processor=object(),
        messages=[],
        history=[],
        text="What is shown?",
        image="image",
        chat_template="template",
        temperature=0.0,
        top_p=0.9,
        max_new_tokens=32,
    )

    assert captured["messages"][0]["role"] == "user"
    assert captured["messages"][0]["content"][0] == {
        "type": "image",
        "image": "image",
    }
    assert captured["chat_template"] == "template"
    assert messages[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "A cat."}],
    }
    assert history == [("[image] What is shown?", "A cat.")]


def test_add_turn_ignores_empty_input(monkeypatch):
    monkeypatch.setattr(
        app,
        "generate_response",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not generate")),
    )

    messages, history = app.add_turn(
        model=object(),
        processor=object(),
        messages=[],
        history=[],
        text=" ",
        image=None,
        chat_template=None,
        temperature=0.0,
        top_p=None,
        max_new_tokens=32,
    )

    assert messages == []
    assert history == []


def test_parse_args_does_not_require_gradio():
    args = app.parse_args(["--model-path", "checkpoint", "--device", "cpu"])

    assert args.model_path == "checkpoint"
    assert args.device == "cpu"
