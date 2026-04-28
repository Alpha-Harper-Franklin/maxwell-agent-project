const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const FINAL_PATH = 'F:\\maxwell_agent_project\\downloads\\ELECTRONICSSTUDENT_2025R2_WINX64.zip';
const LOG_PATH = 'F:\\maxwell_agent_project\\logs\\ansys_browser_download.log';
const PAGE_URL = 'https://www.ansys.com/en-gb/academic/students/ansys-electronics-desktop-student';

function log(message) {
  const stamp = new Date().toISOString().replace('T', ' ').replace('Z', '');
  fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });
  fs.appendFileSync(LOG_PATH, `[${stamp}] ${message}\n`, 'utf8');
}

async function clickFirstVisible(page, selectors) {
  for (const selector of selectors) {
    const locator = page.locator(selector).first();
    try {
      if (await locator.isVisible({ timeout: 3000 })) {
        await locator.click({ timeout: 5000 });
        log(`Clicked selector: ${selector}`);
        return true;
      }
    } catch {
    }
  }
  return false;
}

async function main() {
  fs.mkdirSync(path.dirname(FINAL_PATH), { recursive: true });
  if (fs.existsSync(FINAL_PATH)) {
    log(`Final file already exists at ${FINAL_PATH}`);
    return;
  }

  const browser = await chromium.launch({
    channel: 'msedge',
    headless: true
  });

  const context = await browser.newContext({
    acceptDownloads: true,
    viewport: { width: 1440, height: 1400 },
    userAgent:
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
  });

  const page = await context.newPage();

  try {
    log(`Navigating to ${PAGE_URL}`);
    await page.goto(PAGE_URL, { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForLoadState('networkidle', { timeout: 120000 }).catch(() => {});

    await clickFirstVisible(page, [
      'button:has-text("Accept")',
      'button:has-text("Allow all")',
      'button:has-text("I agree")',
      'text=Accept All Cookies',
      'text=Accept cookies'
    ]);

    const downloadLocators = [
      page.getByRole('button', { name: /download ansys electronics desktop student/i }),
      page.getByRole('link', { name: /download ansys electronics desktop student/i }),
      page.locator('text=/Download Ansys Electronics Desktop Student/i'),
      page.locator('a:has-text("Download Ansys Electronics Desktop Student")'),
      page.locator('button:has-text("Download Ansys Electronics Desktop Student")'),
      page.locator('a[href*="ELECTRONICSSTUDENT_2025R2_WINX64"]')
    ];

    let downloadTriggered = false;
    for (const locator of downloadLocators) {
      try {
        await locator.first().scrollIntoViewIfNeeded({ timeout: 5000 });
        const download = await Promise.race([
          page.waitForEvent('download', { timeout: 15000 }),
          (async () => {
            await locator.first().click({ timeout: 10000 });
            return null;
          })()
        ]);
        if (download) {
          log(`Download event fired from ${await download.url()}`);
          await download.saveAs(FINAL_PATH);
          log(`Download saved to ${FINAL_PATH}`);
          downloadTriggered = true;
          break;
        }
      } catch (error) {
        log(`Download trigger attempt failed: ${error.message}`);
      }
    }

    if (!downloadTriggered) {
      const link = page.locator('a[href*="ELECTRONICSSTUDENT_2025R2_WINX64"], a[href*="release2025R2"], a[href$=".zip"]').first();
      if (await link.count()) {
        const href = await link.getAttribute('href');
        log(`Falling back to href candidate: ${href}`);
        await link.scrollIntoViewIfNeeded({ timeout: 5000 }).catch(() => {});
        const downloadPromise = page.waitForEvent('download', { timeout: 30000 });
        await link.click({ timeout: 10000, force: true });
        const download = await downloadPromise;
        log(`Download event fired from fallback ${await download.url()}`);
        await download.saveAs(FINAL_PATH);
        log(`Download saved to ${FINAL_PATH}`);
        downloadTriggered = true;
      }
    }

    if (!downloadTriggered) {
      throw new Error('Could not trigger Ansys student download.');
    }
  } finally {
    await context.close().catch(() => {});
    await browser.close().catch(() => {});
  }
}

main().catch((error) => {
  log(`Download script failed: ${error.stack || error.message}`);
  process.exit(1);
});
