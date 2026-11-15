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
  await page.locator("#import-file").setInputFiles({
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

test("automatic audit shows both judgments immediately, filters, corrects and exports without two reviewers", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await reviewer(page, "Bob");
  const automatic = (id, bucket, confidence, status = "labeled") => ({
    id,
    text: "The bank is nearby.",
    target: { start: 4, end: 8, text: "bank" },
    candidates: [
      { id: "finance", definition: "A financial institution" },
      { id: "seat", definition: "A long seat" },
    ],
    language: "en",
    definition_language: "en",
    gold_ids:
      status === "labeled"
        ? [bucket === "jev_high_confidence" ? "finance" : "seat"]
        : null,
    provenance: {
      source: "offline automatic fixture",
      document_id: id,
      generation: {
        difficulty: "hard",
        area: "linguistics",
        paper_id: "paper-browser",
      },
      automatic: {
        status,
        bucket,
        selected_by:
          status === "labeled"
            ? bucket === "jev_high_confidence"
              ? "jev"
              : "codex"
            : null,
        threshold: 0.8,
        label_tier: "silver",
        jev: {
          choice: "finance",
          selected_ids: ["finance"],
          confidence,
          probabilities: {
            finance: confidence,
            seat: Math.max(0, 1 - confidence - 0.05),
          },
          none_probability: 0.025,
          insufficient_probability: 0.025,
        },
        codex: {
          outcome: status === "labeled" ? "selected" : "insufficient_context",
          selected_ids: status === "labeled" ? ["seat"] : [],
          rationale: "Offline Codex rationale: seating context.",
        },
        escalation_reason:
          bucket === "jev_high_confidence" ? null : "jev_low_confidence",
      },
    },
  });
  const rows = [
    automatic("auto-browser", "jev_low_codex_disagreed", 0.6),
    automatic("auto-unresolved", "unresolved", 0.3, "unresolved"),
    automatic("auto-high", "jev_high_confidence", 0.95),
  ];
  await page.locator("#import-file").setInputFiles({
    name: "automatic.jsonl",
    mimeType: "application/x-ndjson",
    buffer: Buffer.from(rows.map((row) => JSON.stringify(row)).join("\n")),
  });
  await expect(page.locator("#notice")).toContainText(
    "Automatic proposals are visible",
  );
  await page.locator("#random-audit").click();
  await expect(page.locator("#notice")).toContainText(
    "Random unreviewed high-confidence",
  );
  await expect(page.locator("#bucket-filter")).toHaveValue(
    "jev_high_confidence",
  );
  await page.locator("#status-filter").selectOption("");
  await page.locator("#bucket-filter").selectOption("jev_low_codex_disagreed");
  await expect(page.locator("#queue .queue-item")).toHaveCount(1);
  await expect(page.locator("#automatic-evidence")).toContainText(
    "Jev judgment",
  );
  await expect(page.locator("#automatic-evidence")).toContainText("60.0%");
  await expect(page.locator("#automatic-evidence")).toContainText(
    "Offline Codex rationale",
  );
  await expect(page.locator("#automatic-evidence")).toContainText(
    "Original proposed IDs: seat",
  );
  await expect(page.locator("#legacy-assistance")).toBeHidden();
  await expect(page.locator("input[name=candidate][value=seat]")).toBeChecked();
  await page.locator("#approve-result").click();
  await expect(page.locator("#vote-status")).toContainText("approved");
  await expect(page.locator("#automatic-dashboard")).toContainText("0.0%");
  await page.locator("#status-filter").selectOption("unreviewed");
  await expect(page.locator("#queue .queue-item")).toHaveCount(0);
  await page.locator("#status-filter").selectOption("");
  await page.locator("#bucket-filter").selectOption("unresolved");
  await expect(page.locator("#approve-result")).toBeDisabled();
  await page.locator("input[name=candidate][value=finance]").check();
  await page.locator("input[name=candidate][value=seat]").check();
  await page
    .locator("#review-note")
    .fill("Both meanings acceptable in this passage.");
  await page.locator("#save-vote").click();
  await expect(page.locator("#vote-status")).toContainText("corrected");
  await page.locator("#difficulty-filter").selectOption("hard");
  await page.locator("#area-filter").selectOption("linguistics");
  await page.locator("#paper-filter").selectOption("paper-browser");
  await expect(page.locator("#queue .queue-item")).toHaveCount(1);
  const goldWait = page.waitForEvent("download");
  await page.locator("#export-button").click();
  const gold = await fs.readFile(await (await goldWait).path(), "utf8");
  const exported = gold.trim().split("\n").map(JSON.parse);
  expect(exported.find((row) => row.id === "auto-browser").gold_ids).toEqual([
    "seat",
  ]);
  expect(exported.find((row) => row.id === "auto-unresolved").gold_ids).toEqual(
    ["finance", "seat"],
  );
  await page.locator("input[name=outcome][value=insufficient_context]").check();
  await page.locator("#save-vote").click();
  await expect(page.locator("#vote-status")).toContainText(
    "insufficient context",
  );
  await page.locator("#reject-result").click();
  await expect(page.locator("#vote-status")).toContainText("rejected");
  const allWait = page.waitForEvent("download");
  await page.locator("#export-all-button").click();
  const all = (await fs.readFile(await (await allWait).path(), "utf8"))
    .trim()
    .split("\n")
    .map(JSON.parse);
  const rejected = all.find((row) => row.id === "auto-unresolved");
  expect(rejected.gold_ids).toBeNull();
  expect(rejected.provenance.review.status).toBe("rejected");
  expect(rejected.provenance.automatic.codex.rationale).toContain(
    "Offline Codex",
  );
  await page.locator("#settings-button").click();
  await page.locator("#api-key").fill("browser-fixture-key");
  await page.locator("#save-to-config").check();
  await page.locator("#settings-form").evaluate((form) => form.requestSubmit());
  await expect(page.locator("#notice")).toContainText("owner-only config");
  await page.reload();
  await page.locator("#settings-button").click();
  await expect(page.locator("#api-key")).toHaveValue("");
  await expect(page.locator("#credential-status")).toContainText(
    "local config",
  );
  await page.locator("#clear-key").click();
  await page.locator("#close-settings").click();
  await page.locator("#bucket-filter").selectOption("jev_low_codex_disagreed");
  await expect(page.locator("#vote-status")).toContainText("approved");
  await page.screenshot({
    path: "../.annotation/automatic-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  expect(errors).toEqual([]);
});
