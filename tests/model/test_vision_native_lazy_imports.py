"""Check native vision dispatch in a fresh interpreter."""

import subprocess
import sys
import textwrap


def test_native_vision_auto_defers_model_imports() -> None:
    script = textwrap.dedent("""
        import sys
        from transformers import CLIPVisionConfig
        from transformers.utils.import_utils import _LazyModule
        import tinyllava.model.vision_tower as vision
        from tinyllava.model.vision_tower import AutoVisionTowerModel

        assert isinstance(vision, _LazyModule)
        deferred = (
            "transformers.models.clip.modeling_clip",
            "transformers.models.dinov2.modeling_dinov2",
            "tinyllava.model.vision_tower.mof.modeling_mof",
        )
        assert len(AutoVisionTowerModel._model_mapping) > 1
        assert all(name not in sys.modules for name in deferred)
        config = CLIPVisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, image_size=16, patch_size=8,
        )
        model = AutoVisionTowerModel.from_config(config)
        assert type(model).__name__ == "CLIPVisionModel"
        assert deferred[0] in sys.modules
        assert all(name not in sys.modules for name in deferred[1:])
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
