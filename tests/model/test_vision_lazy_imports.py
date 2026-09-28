import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("model_type", ["clip", "mof"])
def test_vision_imports_load_only_the_selected_model(model_type):
    # pytest collection imports model classes; a fresh interpreter catches
    # accidental eager imports that in-process sys.modules checks would miss.
    script = textwrap.dedent("""
        import sys
        from transformers import CLIPVisionConfig, Dinov2Config
        from transformers.utils.import_utils import _LazyModule
        import tinyllava.model.vision_tower as vision
        import tinyllava.model.vision_tower.mof as mof
        import tinyllava.model.connector.mof as connector
        from tinyllava.model.vision_tower import AutoVisionTowerModel, MofVisionConfig
        from tinyllava.model.connector.mof import MofConnectorConfig

        assert all(isinstance(module, _LazyModule) for module in (vision, mof, connector))
        clip = CLIPVisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, image_size=16, patch_size=8,
        )
        config = MofVisionConfig(
            clip_config=clip,
            dinov2_config=Dinov2Config(
                hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                image_size=16, patch_size=8,
            ),
        )
        MofConnectorConfig()
        deferred = (
            "tinyllava.model.vision_tower.mof.modeling_mof",
            "tinyllava.model.connector.mof.modeling_mof",
            "transformers.models.clip.modeling_clip",
            "transformers.models.dinov2.modeling_dinov2",
        )
        assert len(AutoVisionTowerModel._model_mapping) > 1
        assert all(name not in sys.modules for name in deferred)
        selected = sys.argv[1]
        model = AutoVisionTowerModel.from_config(config if selected == "mof" else clip)
        assert (deferred[0] in sys.modules) == (selected == "mof")
        assert deferred[1] not in sys.modules
        assert deferred[2] in sys.modules
        assert (deferred[3] in sys.modules) == (selected == "mof")
        assert type(model).__name__ == ("MofVisionModel" if selected == "mof" else "CLIPVisionModel")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, model_type],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
