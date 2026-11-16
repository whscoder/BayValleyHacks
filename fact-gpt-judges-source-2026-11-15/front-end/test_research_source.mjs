import assert from "node:assert/strict";
import { researchText } from "./backendClient.js";

const originalFetch = globalThis.fetch;
const requests = [];
globalThis.fetch = async (_url, init) => {
  requests.push(JSON.parse(init.body));
  return new Response(JSON.stringify({ status: "researched" }), {
    status: 200,
    headers: { "Content-Type": "application/json" }
  });
};
try {
  await researchText("Article text", "Title", null, null, undefined, "https://cnn.com/story");
  assert.equal(requests[0].source_url, "https://cnn.com/story");
  await researchText("Legacy caller", "Title");
  assert.equal(Object.hasOwn(requests[1], "source_url"), false);
} finally {
  globalThis.fetch = originalFetch;
}
