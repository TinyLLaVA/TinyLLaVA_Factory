# 维护文档

本站使用 Zensical 和 mkdocstrings。英文页面位于 `docs/en`，中文页面位于 `docs/zh`；导航分别配置在 `zensical.toml` 和 `zensical.zh.toml` 中。

## 构建与预览

使用项目环境中的 Python 3.11 或更新版本，在仓库根目录运行：

```bash
python -m pip install -e ".[docs]"
python docs/build.py
python -m http.server 8000 --bind 127.0.0.1 --directory site
```

`docs` extra 将文档工具加入项目环境，普通安装仍使用 `pip install -e .`。使用 uv 时运行 `uv sync --locked --extra docs --inexact`，再执行 `uv run --no-sync python docs/build.py`。

英文入口为 `http://localhost:8000/`，中文入口为 `http://localhost:8000/zh/`。SSH 环境需转发 8000 端口。修改页面后重新构建；单语言热重载可用 `zensical serve` 或 `zensical serve -f zensical.zh.toml`。

构建脚本通过 Zensical 的 `--strict` 检查生成两种语言，产物保存在 `site/`。

## 编辑页面与 API 参考

同步维护两种语言的指南与导航。语言菜单打开目标语言首页。

API 页面通过 `:::` 引用 `tinyllava/` 中的符号。显式选择公开成员，并保持中英文页面的符号列表一致。

mkdocstrings 通过 Griffe 读取源码，并在无法静态解析时允许动态导入（`allow_inspection = true`）。运行时生成的 API 可使用针对性的 Griffe 扩展或 `force_inspection`，配置后检查生成的成员与签名。详见[解析选项](https://mkdocstrings.github.io/python/usage/configuration/general/#allow_inspection)。

### Docstring 风格

使用 [Google 风格 docstring](https://mkdocstrings.github.io/griffe/reference/docstrings/#google-style)：先用一句话说明用途，再按需补充 `Args`、`Returns`、`Raises` 和 `Examples`。说明单位、张量形状、mask 规则、副作用和影响使用的默认行为。类型和默认值以 Python 签名为准，缺少类型注解时才在 docstring 中补充类型。dataclass 字段的属性 docstring 紧跟字段声明。

各节之间留一个空行，条目缩进四个空格。使用 Markdown 链接和 Python 代码块，示例采用 TinyLLaVA 的公开接口，并说明需要准备的模型或数据。

API 参考使用 mkdocstrings 内置的类型标识、参数标题、独立签名和列表式说明，两种语言的 Python handler 配置保持一致。配置参考[官方站点](https://github.com/mkdocstrings/mkdocstrings/blob/main/zensical.toml)。

中文正文每段写在一个源文件行内，避免 Markdown 软换行引入空格。保留 `checkpoint`、`tokenizer`、`processor`、`collator` 和 `chat template` 等技术术语。代码块、表格和不同列表项各自保留换行。

## 测试与 CI

安装测试依赖后运行测试；`slow` 单独执行：

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m pytest -m slow
```

默认测试排除 `slow`。慢测试创建随机初始化的微型 Llama/CLIP，以及红、蓝两张纯色图片，用同一个问题提问颜色。测试经训练 CLI 启动真实 worker，保存后重新加载 checkpoint，要求生成的两个单词答案均正确。测试离线使用 CPU，覆盖 `configs/training/` 下的所有预设及种子 0、42；保留各预设的调优策略，包括 `qlora_finetune` 当前使用的 int8 策略，仅为小任务覆盖 batch、学习率、步数和运行设置。

图片放在提示末尾，使仅训练 connector 的 pretrain 也能在冻结随机语言模型时学习。这个过拟合测试验证上述预设和种子的训练、adapter 保存与推理链路。

颜色测试保留预设默认开启的梯度检查点。快速测试还覆盖七种模型预设的缩小版反向传播、保存重载与生成，以及 PEFT 组件训练策略、int8 组件组装、多图 connector 和评测分片。另一个慢测试从本地组件组装 Qwen2/SigLIP，检查相同种子的训练可重复性、LoRA 断点恢复、连续两个 LoRA 阶段及后续全量微调。

`CI` 工作流使用 CPU 版 PyTorch，分别在最低支持的 Transformers 版本和 `pyproject.toml` 允许的最新版本上执行快速测试，并只在最新版本的任务中严格构建一次双语文档。`uv.lock` 用于复现开发环境，不约束 CI 的依赖解析。手动运行时勾选 `slow` 可在两个版本上追加这些训练测试；构建站点保存在 `github-pages` 产物中，保留 14 天。

## 使用 GitHub Actions 发布

发布步骤：

1. 在 `main` 和 `develop` 两个分支同步维护 `.github/workflows/ci.yml`，以启用手动运行。
2. 在 **Settings → Pages → Source** 选择 **GitHub Actions**。
3. 在 `github-pages` 环境的部署规则中允许 `develop`。
4. 打开 **Actions → CI → Run workflow**，选择 `develop` 并勾选 `deploy`。

发布使用同一份已验证的构建产物，两种语言的地址由 GitHub Pages 提供。本地测试部署路径可运行：

```bash
python docs/build.py --base-url https://example.org/project
```

再次运行不带 `--base-url` 的 `docs/build.py`，可恢复 localhost 预览路径。
