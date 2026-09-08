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
    assert history == [
        {"role": "user", "content": "[image] What is shown?"},
        {"role": "assistant", "content": "A cat."},
    ]


def test_add_turn_preserves_message_history(monkeypatch):
    monkeypatch.setattr(app, "generate_response", lambda **kwargs: "Second answer.")
    previous = [
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "First answer."},
    ]
    _, history = app.add_turn(
        model=object(), processor=object(), messages=[], history=previous,
        text="Second question.", image=None, chat_template=None,
        temperature=0.0, top_p=None, max_new_tokens=32,
    )
    assert len(previous) == 2
    assert history == previous + [
        {"role": "user", "content": "Second question."},
        {"role": "assistant", "content": "Second answer."},
    ]


def test_build_demo_and_chatbot_message_format(monkeypatch):
    import gradio as gr

    monkeypatch.setattr(app, "generate_response", lambda **kwargs: "A cat.")
    demo = app.build_demo(model=object(), processor=object())
    try:
        chatbots = [block for block in demo.blocks.values() if isinstance(block, gr.Chatbot)]
        assert len(chatbots) == 1
        _, history = app.add_turn(
            model=object(), processor=object(), messages=[], history=[],
            text="What is shown?", image=None, chat_template=None,
            temperature=0.0, top_p=None, max_new_tokens=32,
        )
        rendered = chatbots[0].postprocess(history)
        assert len(rendered.root) == 2
        assert rendered.root[-1].role == "assistant"
        assert demo.get_config_file()["components"]
    finally:
        demo.close()


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
