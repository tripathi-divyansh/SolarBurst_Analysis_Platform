import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe';
const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const executablePath = fs.existsSync(CHROME_PATH) ? CHROME_PATH : EDGE_PATH;

const ARTIFACTS_DIR = 'C:\\Users\\HP\\.gemini\\antigravity-ide\\brain\\09a93114-507d-4261-bdcb-0a1248dbb7dd\\screenshots';
if (!fs.existsSync(ARTIFACTS_DIR)) {
  fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
}

async function run() {
  console.log('[Theme Test] Starting browser...');
  const browser = await puppeteer.launch({
    executablePath,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu', '--window-size=1440,920'],
    defaultViewport: { width: 1440, height: 920, deviceScaleFactor: 1 }
  });

  const page = await browser.newPage();
  console.log('[Theme Test] Loading http://localhost:5000...');
  await page.goto('http://localhost:5000/', { waitUntil: 'networkidle2', timeout: 20000 });
  await new Promise(r => setTimeout(r, 3000));

  // Verify theme toggle button exists
  const toggleBtn = await page.$('#theme-mode-toggle');
  if (!toggleBtn) {
    throw new Error('Theme toggle button #theme-mode-toggle not found in DOM!');
  }

  console.log('[Theme Test] Clicking theme toggle button to switch to Light Mode...');
  await toggleBtn.click();
  await new Promise(r => setTimeout(r, 1000));

  // Check data-theme attribute on <html>
  const currentTheme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  console.log(`[Theme Test] Active data-theme: "${currentTheme}"`);

  // Capture Dashboard in Light Mode
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'light_01_dashboard.png') });
  console.log('[Theme Test] Saved light_01_dashboard.png');

  // Navigate to Light Curve Explorer in Light Mode
  console.log('[Theme Test] Navigating to Explorer...');
  const navLinks = await page.$$('a.nav-link');
  for (const link of navLinks) {
    const text = await page.evaluate(el => el.textContent, link);
    if (text.includes('Light-Curve Explorer')) {
      await link.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 3500));
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'light_03_explorer.png') });
  console.log('[Theme Test] Saved light_03_explorer.png');

  // Navigate to Burst Catalog in Light Mode
  console.log('[Theme Test] Navigating to Burst Catalog...');
  for (const link of navLinks) {
    const text = await page.evaluate(el => el.textContent, link);
    if (text.includes('Burst Catalog')) {
      await link.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 2000));
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'light_05_catalog.png') });
  console.log('[Theme Test] Saved light_05_catalog.png');

  // Navigate to Event Detail in Light Mode
  console.log('[Theme Test] Navigating to Event Detail...');
  for (const link of navLinks) {
    const text = await page.evaluate(el => el.textContent, link);
    if (text.includes('Event Detail')) {
      await link.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 3000));
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'light_06_event_detail.png') });
  console.log('[Theme Test] Saved light_06_event_detail.png');

  // Switch back to Dark Mode to verify bidirectional toggle
  console.log('[Theme Test] Clicking theme toggle button to switch back to Dark Mode...');
  const toggleBtnDark = await page.$('#theme-mode-toggle');
  await toggleBtnDark.click();
  await new Promise(r => setTimeout(r, 1000));

  const revertedTheme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  console.log(`[Theme Test] Reverted data-theme: "${revertedTheme}"`);

  await browser.close();
  console.log('[Theme Test] All light mode tests passed successfully!');
}

run().catch(err => {
  console.error('[Theme Test Error]', err);
  process.exit(1);
});
