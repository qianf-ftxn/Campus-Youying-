const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function page(responses) {
  const elements = new Map();
  function element(selector) {
    if (!elements.has(selector)) elements.set(selector, {value: '', textContent: '', hidden: true,
      addEventListener() {}, querySelectorAll() { return []; }});
    return elements.get(selector);
  }
  const context = vm.createContext({document: {querySelector: element, querySelectorAll: () => []},
    window: {location: {replace() {}}}, sessionStorage: {getItem: () => JSON.stringify({token: 'test'})},
    fetch: () => new Promise(() => {}), setTimeout: (fn) => fn(), console});
  vm.runInContext(fs.readFileSync(path.resolve(__dirname, '../app/static/student.js'), 'utf8'), context);
  vm.runInContext('state.profile = {id: 1}', context);
  context.fetch = async () => ({ok: true, json: async () => responses.shift()});
  return {context, label: element('#memoryState')};
}

test('Student sees background update status, never memory payload', async () => {
  const p = page([{state: 'UPDATING', memories: ['private-text']}, {state: 'READY', content: 'private-text'}]);
  await vm.runInContext('refreshMemoryStatus()', p.context);
  assert.equal(p.label.textContent, '记忆已更新');
  assert.equal(p.label.hidden, false);
  assert.equal(p.label.textContent.includes('private-text'), false);
});

test('Disabled memory is hidden; vector outage reports synchronization state', async () => {
  const p = page([{state: 'DISABLED'}]);
  await vm.runInContext('refreshMemoryStatus()', p.context);
  assert.equal(p.label.hidden, true);
  p.context.fetch = async () => ({ok: true, json: async () => ({state: 'DEGRADED'})});
  await vm.runInContext('refreshMemoryStatus()', p.context);
  assert.equal(p.label.textContent, '记忆已保存，检索同步待完成');
});

test('Background index recovery changes the indicator without a page refresh', async () => {
  const p = page([{state: 'DEGRADED'}, {state: 'READY'}]);
  await vm.runInContext('refreshMemoryStatus()', p.context);
  assert.equal(p.label.textContent, '记忆已更新');
});
