from types import SimpleNamespace
from unittest.mock import Mock

from tinyllava.serve import chat


def setup_chat(monkeypatch):
    bundle = SimpleNamespace(model=Mock(), processor=object())
    load = Mock(return_value=bundle)
    monkeypatch.setattr(chat, "load_tinyllava_checkpoint_bundle", load)
    monkeypatch.setattr(chat, "load_images", lambda paths: list(paths))
    monkeypatch.setattr(
        chat,
        "make_user_message",
        lambda text, images: {
            "role": "user",
            "content": text,
            "images": images,
        },
    )
    generate = Mock(side_effect=["first answer", "second answer", "fresh answer"])
    monkeypatch.setattr(chat, "generate_response", generate)
    return load, generate


def test_chat_retains_history_images_and_resets(monkeypatch):
    load, generate = setup_chat(monkeypatch)
    inputs = iter(["follow up", "/reset", "new topic", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    chat.run_chat(
        model_path="checkpoint", device="cpu", query="describe", image_files=["one.png"]
    )
    load.assert_called_once()
    second = generate.call_args_list[1].kwargs["messages"]
    assert [message["role"] for message in second] == ["user", "assistant", "user"]
    assert second[0]["images"] == ["one.png"]
    assert second[1]["content"][0]["text"] == "first answer"
    assert second[2]["images"] == []
    assert len(generate.call_args_list[2].kwargs["messages"]) == 1


def test_single_turn_generates_once_without_reading_input(monkeypatch):
    _, generate = setup_chat(monkeypatch)
    monkeypatch.setattr(
        "builtins.input", Mock(side_effect=AssertionError("unexpected input"))
    )
    chat.run_chat(
        model_path="checkpoint", device="cpu", query="hello", interactive=False
    )
    generate.assert_called_once()


def test_interactive_image_command_and_eof(monkeypatch):
    _, generate = setup_chat(monkeypatch)
    read = Mock(side_effect=["/image photo.png", "describe", EOFError])
    monkeypatch.setattr("builtins.input", read)
    chat.run_chat(model_path="checkpoint", device="cpu")
    assert generate.call_args.kwargs["messages"][0]["images"] == ["photo.png"]
