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

构建脚本生成两种语言，并检查页面、API 锚点、语言链接和本地资源。生成产物保存在 `site/`。

## 编辑页面与 API 参考

同步维护两种语言的指南与导航。语言菜单打开目标语言首页。

API 页面通过 `:::` 引用 `tinyllava/` 中的符号。显式选择公开成员，并保持中英文页面的符号列表一致。

mkdocstrings 通过 Griffe 读取源码，并在无法静态解析时允许动态导入（`allow_inspection = true`）。运行时生成的 API 可使用针对性的 Griffe 扩展或 `force_inspection`，配置后检查生成的成员与签名。详见[解析选项](https://mkdocstrings.github.io/python/usage/configuration/general/#allow_inspection)。

### Docstring 风格

使用 [Google 风格 docstring](https://mkdocstrings.github.io/griffe/reference/docstrings/#google-style)：先用一句话说明用途，再按需补充 `Args`、`Returns`、`Raises` 和 `Examples`。说明单位、张量形状、mask 规则、副作用和影响使用的默认行为。类型和默认值以 Python 签名为准，缺少类型注解时才在 docstring 中补充类型。dataclass 字段放在 `Attributes` 中，名称与源码一致。

各节之间留一个空行，条目缩进四个空格。使用 Markdown 链接和 Python 代码块，示例采用 TinyLLaVA 的公开接口，并说明需要准备的模型或数据。

API 参考使用 mkdocstrings 内置的类型标识、参数标题、独立签名和列表式说明，两种语言的 Python handler 配置保持一致。配置参考[官方站点](https://github.com/mkdocstrings/mkdocstrings/blob/main/zensical.toml)。

中文正文每段写在一个源文件行内，避免 Markdown 软换行引入空格。保留 `checkpoint`、`tokenizer`、`processor`、`collator` 和 `chat template` 等技术术语。代码块、表格和不同列表项各自保留换行。

## 使用 GitHub Actions 发布

`Documentation` 工作流检查面向 `develop` 的相关 PR 与推送，保留 `tinyllava-docs` 预览产物 14 天。

发布步骤：

1. 将工作流集成到仓库默认分支，以启用手动运行。
2. 在 **Settings → Pages → Source** 选择 **GitHub Actions**。
3. 在 `github-pages` 环境的部署规则中允许 `develop`。
4. 打开 **Actions → Documentation → Run workflow**，选择 `develop` 并勾选 `deploy`。

发布时根据 GitHub Pages 返回的地址配置两种语言。本地测试部署路径可运行：

```bash
python docs/build.py --base-url https://example.org/project
```

再次运行不带 `--base-url` 的 `docs/build.py`，可恢复 localhost 预览路径。
