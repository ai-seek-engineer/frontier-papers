# Frontier Papers userscripts

## 公式增强复制

将 [frontier-papers-math-copy.user.js](./frontier-papers-math-copy.user.js) 的完整内容粘贴到 Tampermonkey 的“添加新脚本”页面并保存。脚本只会在 `https://ai-seek-engineer.github.io/frontier-papers/*` 生效。

使用方式：拖选一段包含公式的正文。被跨过的 MathJax 公式会高亮，随后可按 `Ctrl/Cmd+C`，或点击选区旁的“复制含 N 个公式的选区”按钮。行内公式会复制为 `$...$`，显示公式会复制为 `\\[...\\]`；不含公式的普通选区仍由浏览器原生复制处理。
