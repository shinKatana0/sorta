// Runs `sorta/web/app/app.js` against a DOM stub and reports whether it reached the end
// and wired its last handlers. Called by tests/test_the_page_script_reaches_its_end.py.
//
// Why an execution probe and not a static read: a `var` assigned near the bottom of the
// file is `undefined` to anything that runs above it, and a boot action placed too early
// threw and killed the rest of the script — every handler below it, Browse and Start
// among them, was silently never wired. Nothing that reads the text can see that.
const fs = require("fs");
const path = process.argv[2] || "C:/repo/sorta/sorta/web/app/app.js";
const src = fs.readFileSync(path, "utf8");

const made = new Set();
function el(id) {
  return {
    id, style: {}, dataset: {},
    classList: {
      toggle() {}, add() {}, remove() {},
      contains: (c) => c === "active" && id === "tab-overview",
    },
    addEventListener(ev) { made.add(id + ":" + ev); },
    appendChild() {}, setAttribute() {}, removeAttribute() {}, remove() {},
    querySelector: () => el("q"), querySelectorAll: () => [],
    textContent: "", innerHTML: "", value: "", checked: false,
    disabled: false, hidden: false,
    focus() {}, click() {}, closest: () => el("c"),
    insertBefore() {}, replaceChildren() {},
  };
}

global.window = {
  I18N: new Proxy({}, { get: () => "" }),
  addEventListener() {},
  location: { href: "", search: "" },
  localStorage: { getItem: () => null, setItem() {} },
  setTimeout: () => 0, setInterval: () => 0, clearInterval() {}, clearTimeout() {},
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  requestAnimationFrame: () => 0, alert() {}, confirm: () => true, open() {},
};
global.document = {
  getElementById: el, createElement: el, createTextNode: () => ({}),
  querySelector: () => el("q"), querySelectorAll: () => [], addEventListener() {},
  body: el("body"), documentElement: el("html"),
};
global.fetch = () => new Promise(() => {});   // never settles: no async path runs
global.setTimeout = () => 0;
global.setInterval = () => 0;
global.clearInterval = () => {};
global.clearTimeout = () => {};
global.localStorage = global.window.localStorage;
global.navigator = { language: "en" };

try {
  new Function(src)();
  console.log("script reached the end");
} catch (e) {
  console.log("SCRIPT DIED");
  console.log(e.stack);
}
console.log("browse handler wired:", made.has("process-browse-btn:click"));
console.log("start handler wired:", made.has("process-start-btn:click"));
