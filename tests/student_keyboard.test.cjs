const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

function keyboard() {
  const elements = new Map();
  let submissions = 0;
  function element(selector) {
    if (!elements.has(selector)) elements.set(selector, {
      value: "", listeners: {}, addEventListener(name, callback) { this.listeners[name] = callback; },
      querySelectorAll() { return []; }, requestSubmit() { submissions++; }
    });
    return elements.get(selector);
  }
  const context = vm.createContext({
    document: { querySelector: element, querySelectorAll: () => [] },
    window: { location: { replace() {} } }, sessionStorage: { getItem: () => null },
    fetch: () => new Promise(() => {}), console
  });
  vm.runInContext(fs.readFileSync(path.resolve(__dirname, "../app/static/student.js"), "utf8"), context);
  vm.runInContext("state.profile = {id: 1}", context);
  const input = element("#messageInput");
  input.value = "待发送消息";
  return {
    input, context, submissions: () => submissions,
    press(overrides = {}) {
      let prevented = false;
      input.listeners.keydown({ key: "Enter", preventDefault() { prevented = true; }, ...overrides });
      return prevented;
    }
  };
}

test("Enter submits once and prevents a newline", () => {
  const page = keyboard();
  assert.equal(page.press(), true);
  assert.equal(page.submissions(), 1);
});
test("Shift+Enter keeps the native newline", () => {
  const page = keyboard();
  assert.equal(page.press({ shiftKey: true }), false);
  assert.equal(page.submissions(), 0);
});
test("IME confirmation does not submit, including Safari keyCode 229", () => {
  const page = keyboard();
  assert.equal(page.press({ isComposing: true }), false);
  assert.equal(page.press({ keyCode: 229 }), false);
  page.input.listeners.compositionstart();
  assert.equal(page.press(), false);
  page.input.listeners.compositionend();
  page.press();
  assert.equal(page.submissions(), 1);
});
test("Empty text, held Enter and busy states do not duplicate submissions", () => {
  const page = keyboard();
  page.input.value = "   "; page.press();
  page.input.value = "测试"; page.press({ repeat: true });
  vm.runInContext("state.sending = true", page.context); page.press();
  vm.runInContext("state.sending = false; state.loadingSession = true", page.context); page.press();
  assert.equal(page.submissions(), 0);
});
