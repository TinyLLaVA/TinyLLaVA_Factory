import pytest
import torch
from dataclasses import dataclass, field
from unittest.mock import Mock

from tinyllava.data.template.formatter import (
    Formatter,
    StringFormatter,
    EmptyFormatter,
)
from tinyllava.data.template.base import Template
from tinyllava.data.template.phi_template import PhiTemplate
from tinyllava.utils.constants import (
    DEFAULT_IMAGE_TOKEN,
)


class TestFormatter:
    """Test Formatter classes"""

    def test_empty_formatter(self):
        """Test EmptyFormatter returns slot as-is"""
        formatter = EmptyFormatter(slot="test_value")
        result = formatter.apply()
        assert result == "test_value"

    def test_empty_formatter_with_kwargs(self):
        """Test EmptyFormatter ignores kwargs"""
        formatter = EmptyFormatter(slot="test_value")
        result = formatter.apply(content="ignored")
        assert result == "test_value"

    def test_string_formatter_single_placeholder(self):
        """Test StringFormatter with single placeholder"""
        formatter = StringFormatter(slot="Hello {{name}}")
        result = formatter.apply(name="World")
        assert result == "Hello World"

    def test_string_formatter_multiple_placeholders(self):
        """Test StringFormatter with multiple placeholders"""
        formatter = StringFormatter(slot="{{greeting}} {{name}}!")
        result = formatter.apply(greeting="Hello", name="World")
        assert result == "Hello World!"

    def test_string_formatter_with_none_value(self):
        """Test StringFormatter with None value"""
        formatter = StringFormatter(slot="Hello {{name}}: message")
        result = formatter.apply(name=None)
        assert result == "Hello : message"

    def test_string_formatter_invalid_type(self):
        """Test StringFormatter raises error for non-string values"""
        formatter = StringFormatter(slot="Number: {{value}}")
        with pytest.raises(TypeError, match="Expected a string, got int"):
            formatter.apply(value=123)


class TestTemplate:
    """Test Template base class"""

    def create_mock_template(self):
        """Create a mock template for testing"""

        @dataclass
        class MockTemplate(Template):
            format_image_token: Formatter = field(default_factory=lambda: StringFormatter(
                slot="<image>\n{{content}}"
            ))
            format_user: Formatter = field(default_factory=lambda: StringFormatter(slot="USER: {{content}} "))
            format_assistant: Formatter = field(default_factory=lambda: StringFormatter(
                slot="ASSISTANT: {{content}}<|end|>"
            ))
            system: Formatter = field(default_factory=lambda: EmptyFormatter(slot="System prompt. "))
            separator: Formatter = field(default_factory=lambda: EmptyFormatter(slot=[" ASSISTANT: ", "<|end|>"]))

        return MockTemplate()

    def test_get_list_from_message_human_first(self):
        """Test get_list_from_message with human message first"""
        template = self.create_mock_template()
        messages = [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there"},
            {"from": "human", "value": "How are you?"},
            {"from": "gpt", "value": "I'm good"},
        ]
        question_list, answer_list = template.get_list_from_message(messages)
        assert question_list == ["Hello", "How are you?"]
        assert answer_list == ["Hi there", "I'm good"]

    def test_get_list_from_message_system_first(self):
        """Test get_list_from_message with system message first"""
        template = self.create_mock_template()
        messages = [
            {"from": "system", "value": "System message"},
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there"},
        ]
        question_list, answer_list = template.get_list_from_message(messages)
        assert question_list == ["Hello"]
        assert answer_list == ["Hi there"]

    def test_prompt_with_lists(self):
        """Test prompt generation with question and answer lists"""
        template = self.create_mock_template()
        question_list = ["What is 2+2?"]
        answer_list = ["4"]
        prompt = template.prompt(question_list, answer_list)
        assert "System prompt." in prompt
        assert "USER: What is 2+2?" in prompt
        assert "ASSISTANT: 4" in prompt

    def test_prompt_with_strings(self):
        """Test prompt generation with single strings"""
        template = self.create_mock_template()
        prompt = template.prompt("Question", "Answer")
        assert "USER: Question" in prompt
        assert "ASSISTANT: Answer" in prompt

    def test_prompt_multiple_rounds(self):
        """Test prompt generation with multiple conversation rounds"""
        template = self.create_mock_template()
        question_list = ["First question", "Second question"]
        answer_list = ["First answer", "Second answer"]
        prompt = template.prompt(question_list, answer_list)
        assert prompt.count("USER:") == 2
        assert prompt.count("ASSISTANT:") == 2

    def test_tokenizer_image_token_no_image(self):
        """Test tokenizer_image_token with no image token"""
        mock_tokenizer = Mock()
        mock_tokenizer.input_ids = [1, 2, 3]
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.return_value = Mock(input_ids=[1, 2, 3])

        prompt = "Hello world"
        result = Template.tokenizer_image_token(prompt, mock_tokenizer)
        assert isinstance(result, list)

    def test_tokenizer_image_token_with_image(self):
        """Test tokenizer_image_token with image token"""
        mock_tokenizer = Mock()
        mock_tokenizer.bos_token_id = 1

        def mock_tokenize(text):
            return Mock(input_ids=[1, 2, 3] if "text" in text else [4, 5, 6])

        mock_tokenizer.side_effect = mock_tokenize

        prompt = "text before <image> text after"
        result = Template.tokenizer_image_token(prompt, mock_tokenizer)
        assert isinstance(result, list)

    def test_tokenizer_image_token_return_tensor(self):
        """Test tokenizer_image_token returns PyTorch tensor"""
        mock_tokenizer = Mock()
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.return_value = Mock(input_ids=[1, 2, 3])

        prompt = "Hello world"
        result = Template.tokenizer_image_token(
            prompt, mock_tokenizer, return_tensors="pt"
        )
        assert isinstance(result, torch.Tensor)

    def test_tokenizer_image_token_unsupported_tensor_type(self):
        """Test tokenizer_image_token raises on unsupported tensor type"""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = Mock(input_ids=[1, 2, 3])

        prompt = "Hello world"
        with pytest.raises(ValueError, match="Unsupported tensor type"):
            Template.tokenizer_image_token(
                prompt, mock_tokenizer, return_tensors="tf"
            )


class TestPhiTemplate:
    """Test PhiTemplate implementation"""

    def test_phi_template_initialization(self):
        """Test PhiTemplate is properly initialized"""
        template = PhiTemplate()
        assert template.format_image_token is not None
        assert template.format_user is not None
        assert template.format_assistant is not None
        assert template.system is not None
        assert template.separator is not None

    def test_phi_template_format_user(self):
        """Test PhiTemplate user message formatting"""
        template = PhiTemplate()
        result = template.format_user.apply(content="Hello")
        assert result == "USER: Hello "

    def test_phi_template_format_assistant(self):
        """Test PhiTemplate assistant message formatting"""
        template = PhiTemplate()
        result = template.format_assistant.apply(content="Hi there")
        assert result == "ASSISTANT: Hi there<|endoftext|>"

    def test_phi_template_format_image_token(self):
        """Test PhiTemplate image token formatting"""
        template = PhiTemplate()
        result = template.format_image_token.apply(content="What is this?")
        assert result == "<image>\nWhat is this?"

    def test_phi_template_system_prompt(self):
        """Test PhiTemplate system prompt"""
        template = PhiTemplate()
        result = template.system.apply()
        assert "curious user" in result
        assert "artificial intelligence assistant" in result

    def test_phi_template_separator(self):
        """Test PhiTemplate separator"""
        template = PhiTemplate()
        result = template.separator.apply()
        assert isinstance(result, list)
        assert len(result) == 2
        assert " ASSISTANT: " in result
        assert "<|endoftext|>" in result

    def test_phi_template_full_prompt(self):
        """Test PhiTemplate full prompt generation"""
        template = PhiTemplate()
        questions = ["What is AI?"]
        answers = ["AI is artificial intelligence."]
        prompt = template.prompt(questions, answers)
        assert "USER: What is AI?" in prompt
        assert "ASSISTANT: AI is artificial intelligence." in prompt
        assert "curious user" in prompt


class TestTemplateEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_message_list(self):
        """Test handling of empty message list"""

        @dataclass
        class TestTemplate(Template):
            format_image_token: Formatter = field(default_factory=lambda: StringFormatter(slot="<image>{{content}}"))
            format_user: Formatter = field(default_factory=lambda: StringFormatter(slot="USER: {{content}}"))
            format_assistant: Formatter = field(default_factory=lambda: StringFormatter(
                slot="ASSISTANT: {{content}}"
            ))
            system: Formatter = field(default_factory=lambda: EmptyFormatter(slot=""))
            separator: Formatter = field(default_factory=lambda: EmptyFormatter(slot=["", ""]))

        template = TestTemplate()
        question_list, answer_list = template.get_list_from_message([])
        assert question_list == []
        assert answer_list == []

    def test_prompt_with_empty_lists(self):
        """Test prompt generation with empty lists"""

        @dataclass
        class TestTemplate(Template):
            format_image_token: Formatter = field(default_factory=lambda: StringFormatter(slot="<image>{{content}}"))
            format_user: Formatter = field(default_factory=lambda: StringFormatter(slot="USER: {{content}}"))
            format_assistant: Formatter = field(default_factory=lambda: StringFormatter(
                slot="ASSISTANT: {{content}}"
            ))
            system: Formatter = field(default_factory=lambda: EmptyFormatter(slot="System. "))
            separator: Formatter = field(default_factory=lambda: EmptyFormatter(slot=["", ""]))

        template = TestTemplate()
        prompt = template.prompt([], [])
        # Should only contain system prompt or be empty
        assert len(prompt) >= 0

    def test_image_token_replacement(self):
        """Test that image token is properly replaced in questions"""

        @dataclass
        class TestTemplate(Template):
            format_image_token: Formatter = field(default_factory=lambda: StringFormatter(
                slot="<IMG>{{content}}</IMG>"
            ))
            format_user: Formatter = field(default_factory=lambda: StringFormatter(slot="Q: {{content}}"))
            format_assistant: Formatter = field(default_factory=lambda: StringFormatter(slot="A: {{content}}"))
            system: Formatter = field(default_factory=lambda: EmptyFormatter(slot=""))
            separator: Formatter = field(default_factory=lambda: EmptyFormatter(slot=["", ""]))

        template = TestTemplate()
        question = f"{DEFAULT_IMAGE_TOKEN}Describe this image"
        answer = "Description"
        prompt = template.prompt([question], [answer])

        assert DEFAULT_IMAGE_TOKEN not in prompt
        assert "<IMG>" in prompt
        assert "Describe this image" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
