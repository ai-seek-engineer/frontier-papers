// ==UserScript==
// @name         Frontier Papers：复制选区中的 MathJax 公式
// @namespace    https://ai-seek-engineer.github.io/frontier-papers/
// @version      0.2.1
// @description  拖选正文后，把 MathJax 公式作为原始 TeX 一并复制。
// @match        https://ai-seek-engineer.github.io/frontier-papers/*
// @run-at       document-idle
// @grant        none
// ==/UserScript==

(() => {
  'use strict';

  const INSTALL_FLAG = '__frontierPapersMathCopyInstalled__';
  const FORMULA_SELECTOR = 'mjx-container';
  const HIGHLIGHT_CLASS = 'frontier-math-copy-selected';
  const TOOLBAR_ID = 'frontier-math-copy-toolbar';
  const TOAST_ID = 'frontier-math-copy-toast';
  let activeFormulas = [];
  let savedRange = null;
  let selectionSyncPending = false;
  let toolbar;
  let toolbarButton;
  let toast;

  if (window[INSTALL_FLAG]) {
    return;
  }
  window[INSTALL_FLAG] = true;

  function currentRange() {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
      return null;
    }
    return selection.getRangeAt(0);
  }

  function selectedFormulas(range) {
    if (!range) {
      return [];
    }

    return Array.from(document.querySelectorAll(FORMULA_SELECTOR)).filter((formula) => {
      try {
        return range.intersectsNode(formula);
      } catch {
        return false;
      }
    });
  }

  function formulaToTeX(formula) {
    const mathJax = window.MathJax;
    const item = mathJax?.startup?.document?.getMathItemsWithin(formula)?.[0];
    if (!item?.math) {
      return '';
    }

    const tex = item.math.trim();
    return item.display ? `\n\\[\n${tex}\n\\]\n` : `$${tex}$`;
  }

  function addBlockBreaks(fragment) {
    fragment.querySelectorAll('br').forEach((element) => {
      element.replaceWith(document.createTextNode('\n'));
    });

    fragment
      .querySelectorAll('p, div, blockquote, li, h1, h2, h3, h4, h5, h6, pre, tr')
      .forEach((element) => {
        element.before(document.createTextNode('\n'));
        element.after(document.createTextNode('\n'));
      });
  }

  function normaliseText(value) {
    return value
      .replace(/\r\n?/g, '\n')
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n[ \t]+/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }

  function selectionAsTeX(range) {
    const formulas = selectedFormulas(range);
    if (formulas.length === 0) {
      return '';
    }

    const fragment = range.cloneContents();
    const clonedFormulas = Array.from(fragment.querySelectorAll(FORMULA_SELECTOR));

    clonedFormulas.forEach((formula, index) => {
      formula.replaceWith(document.createTextNode(formulaToTeX(formulas[index])));
    });

    fragment.querySelectorAll('mjx-assistive-mml, script, style').forEach((element) => {
      element.remove();
    });
    addBlockBreaks(fragment);

    return normaliseText(fragment.textContent || '');
  }

  function updateHighlights(formulas) {
    const next = new Set(formulas);
    activeFormulas.forEach((formula) => {
      if (!next.has(formula)) {
        formula.classList.remove(HIGHLIGHT_CLASS);
      }
    });
    formulas.forEach((formula) => formula.classList.add(HIGHLIGHT_CLASS));
    activeFormulas = formulas;
  }

  function positionToolbar(range) {
    if (!toolbar || toolbar.hidden) {
      return;
    }

    const rect = range.getBoundingClientRect();
    const margin = 8;
    const toolbarWidth = toolbar.offsetWidth;
    const toolbarHeight = toolbar.offsetHeight;
    const left = Math.max(margin, Math.min(rect.left, window.innerWidth - toolbarWidth - margin));
    const below = rect.bottom + margin;
    const top = below + toolbarHeight <= window.innerHeight ? below : Math.max(margin, rect.top - toolbarHeight - margin);

    toolbar.style.left = `${left}px`;
    toolbar.style.top = `${top}px`;
  }

  function hideToolbar() {
    if (toolbar) {
      toolbar.hidden = true;
    }
  }

  function clearCopiedSelection() {
    savedRange = null;
    updateHighlights([]);
    hideToolbar();
    window.getSelection()?.removeAllRanges();
  }

  function showToolbar(range, formulaCount) {
    toolbarButton.textContent = `复制含 ${formulaCount} 个公式的选区`;
    toolbar.hidden = false;
    positionToolbar(range);
  }

  function showToast(message) {
    if (!toast) {
      return;
    }

    toast.textContent = message;
    toast.hidden = false;
    window.clearTimeout(showToast.timeoutId);
    showToast.timeoutId = window.setTimeout(() => {
      toast.hidden = true;
    }, 1800);
  }

  async function writeClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.setAttribute('aria-hidden', 'true');
      textarea.style.cssText = 'position:fixed;left:-9999px;top:0;';
      document.body.append(textarea);
      textarea.select();
      document.execCommand('copy');
      textarea.remove();
    }
  }

  async function copyRange(range) {
    const text = selectionAsTeX(range);
    const formulas = selectedFormulas(range);
    if (!text || formulas.length === 0) {
      return;
    }

    // Clipboard APIs can resolve slowly in a userscript context.  Dismiss the
    // selection UI before awaiting them so a completed click never leaves a
    // stale toolbar on screen.
    clearCopiedSelection();
    await writeClipboard(text);
    showToast(`已复制，含 ${formulas.length} 个公式`);
  }

  function syncSelection() {
    selectionSyncPending = false;
    const range = currentRange();
    const formulas = selectedFormulas(range);
    updateHighlights(formulas);

    if (!range || formulas.length === 0) {
      savedRange = null;
      hideToolbar();
      return;
    }

    savedRange = range.cloneRange();
    showToolbar(savedRange, formulas.length);
  }

  function scheduleSelectionSync() {
    if (selectionSyncPending) {
      return;
    }

    selectionSyncPending = true;
    window.requestAnimationFrame(syncSelection);
  }

  function onCopy(event) {
    const range = currentRange();
    if (!range || selectedFormulas(range).length === 0) {
      return;
    }

    const text = selectionAsTeX(range);
    if (!text) {
      return;
    }

    event.preventDefault();
    event.clipboardData.setData('text/plain', text);
    event.clipboardData.setData('text/markdown', text);
    showToast(`已复制，含 ${selectedFormulas(range).length} 个公式`);
  }

  function installUi() {
    const style = document.createElement('style');
    style.textContent = `
      .${HIGHLIGHT_CLASS} {
        background: rgba(23, 105, 170, 0.16);
        border-radius: 3px;
        outline: 2px solid rgba(23, 105, 170, 0.72);
        outline-offset: 2px;
      }

      #${TOOLBAR_ID} {
        position: fixed;
        z-index: 2147483647;
        display: flex;
        gap: 8px;
        padding: 6px;
        border: 1px solid #c8d3df;
        border-radius: 7px;
        background: #fff;
        box-shadow: 0 4px 16px rgba(18, 35, 53, 0.18);
      }

      #${TOOLBAR_ID} button {
        border: 0;
        border-radius: 4px;
        padding: 6px 9px;
        color: #fff;
        background: #1769aa;
        font: 600 13px/1.2 system-ui, sans-serif;
        cursor: pointer;
      }

      #${TOAST_ID} {
        position: fixed;
        z-index: 2147483647;
        right: 18px;
        bottom: 18px;
        padding: 8px 12px;
        border-radius: 6px;
        color: #fff;
        background: rgba(32, 36, 42, 0.92);
        font: 13px/1.3 system-ui, sans-serif;
      }
    `;
    document.head.append(style);

    toolbar = document.createElement('div');
    toolbar.id = TOOLBAR_ID;
    toolbar.hidden = true;
    toolbar.setAttribute('role', 'toolbar');

    toolbarButton = document.createElement('button');
    toolbarButton.type = 'button';
    toolbarButton.addEventListener('mousedown', (event) => event.preventDefault());
    toolbarButton.addEventListener('click', () => {
      if (savedRange) {
        copyRange(savedRange);
      }
    });
    toolbar.append(toolbarButton);
    document.body.append(toolbar);

    toast = document.createElement('div');
    toast.id = TOAST_ID;
    toast.hidden = true;
    toast.setAttribute('role', 'status');
    toast.setAttribute('aria-live', 'polite');
    document.body.append(toast);
  }

  function install() {
    installUi();
    document.addEventListener('selectionchange', scheduleSelectionSync);
    document.addEventListener('copy', onCopy);
    window.addEventListener('scroll', () => {
      if (savedRange) {
        positionToolbar(savedRange);
      }
    }, true);
    window.addEventListener('resize', () => {
      if (savedRange) {
        positionToolbar(savedRange);
      }
    });
    console.info('[Frontier Papers Math Copy] 已启用');
  }

  function waitForMathJax(attempt = 0) {
    const mathJax = window.MathJax;
    if (mathJax?.startup?.document && mathJax?.startup?.promise) {
      mathJax.startup.promise.then(install);
      return;
    }

    if (attempt < 100) {
      window.setTimeout(() => waitForMathJax(attempt + 1), 100);
    } else {
      console.warn('[Frontier Papers Math Copy] 未找到 MathJax，脚本未启用');
    }
  }

  waitForMathJax();
})();
