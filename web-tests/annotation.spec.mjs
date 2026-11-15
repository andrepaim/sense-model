import { test, expect } from "@playwright/test";
import fs from "node:fs/promises";
import crypto from "node:crypto";
async function reviewer(page, name) {
  await page.locator("#reviewer-name").fill(name);
  await page.locator("#reviewer-form").evaluate((f) => f.requestSubmit());
  await expect(page.locator("#notice")).toContainText(`Reviewing as ${name}`);
}
async function key(page, model = "jev-latest") {
  await page.locator("#settings-button").click();
  await page.locator("#api-key").fill("browser-fixture-key");
  await page.locator("#jev-model").fill(model);
  await page.locator("#settings-form").evaluate((f) => f.requestSubmit());
  await expect(page.locator("#settings-dialog")).not.toBeVisible();
}

test("blind review, Jev failure/suggestion, revision, second review, adjudication, exact export and mobile", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await reviewer(page, "Alice");
  const text = "😀 The <img src=x onerror=alert(1)> bank is nearby.";
  const start = Array.from(text.slice(0, text.indexOf("bank"))).length;
  const record = {
    id: "browser-real",
    text,
    target: { start, end: start + 4, text: "bank" },
    language: "en",
    definition_language: "en",
    candidates: [
      { id: "finance", definition: "A financial institution" },
      { id: "seat", definition: "A long seat" },
    ],
    gold_ids: ["seat"],
    provenance: {
      source: "browser fixture",
      document_id: "browser-doc",
      source_url: "https://example.org/document",
      revision: "1",
      license: "CC0-1.0",
      license_url: "https://creativecommons.org/publicdomain/zero/1.0/",
      domain: "general",
      lemma: "bank",
    },
  };
  await page
    .locator("#import-file")
    .setInputFiles({
      name: "candidates.jsonl",
      mimeType: "application/x-ndjson",
      buffer: Buffer.from(JSON.stringify(record) + "\n"),
    });
  await expect(page.locator("#sentence mark")).toHaveText("bank");
  await expect(page.locator("#sentence img")).toHaveCount(0);
  await expect(page.locator("#assist-button")).toBeDisabled();
  await page.locator("input[name=candidate][value=finance]").check();
  await page.locator("#save-vote").click();
  await expect(page.locator("#vote-status")).toContainText("saved");
  await key(page, "fixture-failure");
  await page.locator("#assist-button").click();
  await expect(page.locator("#notice")).toContainText("connection failed");
  await expect(
    page.locator("input[name=candidate][value=finance]"),
  ).toBeChecked();
  await key(page);
  await page.locator("#assist-button").click();
  await expect(page.locator("#ai-results")).toContainText("jev-fixture-1");
  await expect(page.locator("#vote-status")).toContainText("AI-exposed");
  await page.locator("input[name=outcome][value=none_fits]").check();
  await page.locator("#save-vote").click();
  await expect(page.locator("#notice")).toContainText("Review saved");
  await page.reload();
  await expect(
    page.locator("input[name=outcome][value=none_fits]"),
  ).toBeChecked();
  expect(await page.evaluate(() => Object.keys(localStorage))).toEqual([]);
  await reviewer(page, "Bob");
  await expect(page.locator("#ai-results")).toBeHidden();
  await expect(page.locator("input[name=candidate]:checked")).toHaveCount(0);
  await page.locator("#adjudication-mode").check();
  await expect(page.locator("#notice")).toContainText("independent vote");
  await expect(page.locator("#adjudication-mode")).not.toBeChecked();
  await page.locator("input[name=candidate][value=finance]").check();
  await page.locator("#save-vote").click();
  await expect(page.locator("#notice")).toContainText("Review saved");
  await page.locator("#adjudication-mode").check();
  await expect(page.locator("#other-reviews")).toContainText("Alice");
  await expect(page.locator("#other-reviews")).toContainText(
    "AI assistance was viewed",
  );
  await page
    .locator("#review-note")
    .fill(
      "Both initial votes selected the financial sense; resolve Alice revision.",
    );
  await page.locator("#finalize-button").click();
  await expect(page.locator("#notice")).toContainText("Adjudication saved");
  const downloads = [];
  page.on("download", (d) => downloads.push(d));
  await page.locator("#export-button").click();
  await expect.poll(() => downloads.length).toBe(2);
  const content = await fs.readFile(await downloads[0].path(), "utf8");
  const manifest = JSON.parse(
    await fs.readFile(await downloads[1].path(), "utf8"),
  );
  expect(JSON.parse(content).gold_ids).toEqual(["finance"]);
  expect(manifest.sha256).toBe(
    crypto.createHash("sha256").update(content).digest("hex"),
  );
  expect(content).not.toContain("browser-fixture-key");
  await page.locator("#settings-button").click();
  await expect(page.locator("#api-key")).toHaveValue("");
  await page.locator("#clear-key").click();
  await expect(page.locator("#credential-status")).toContainText("No API key");
  await page.locator("#close-settings").click();
  await page.screenshot({
    path: "../.annotation/browser-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  expect(errors).toEqual([]);
});
