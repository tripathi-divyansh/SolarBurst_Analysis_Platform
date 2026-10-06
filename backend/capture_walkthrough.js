import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe';
const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

const executablePath = fs.existsSync(CHROME_PATH) ? CHROME_PATH : EDGE_PATH;
console.log(`[Interactive Tour] Using browser: ${executablePath}`);

const ARTIFACTS_DIR = 'C:\\Users\\HP\\.gemini\\antigravity-ide\\brain\\09a93114-507d-4261-bdcb-0a1248dbb7dd\\screenshots';
if (!fs.existsSync(ARTIFACTS_DIR)) {
  fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
}

async function run() {
  const browser = await puppeteer.launch({
    executablePath,
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--window-size=1440,920'
    ],
    defaultViewport: {
      width: 1440,
      height: 920,
      deviceScaleFactor: 1
    }
  });

  const page = await browser.newPage();

  console.log('[Tour] Loading http://localhost:5000...');
  await page.goto('http://localhost:5000/', { waitUntil: 'networkidle2', timeout: 20000 });
  // Wait for initial load hook to fetch datasets and completed results
  await new Promise(r => setTimeout(r, 3000));

  const STEPS = [
    { name: '01_dashboard.png', linkText: 'Dashboard', waitMs: 1500 },
    { name: '02_upload_inspect.png', linkText: 'Upload & Inspect', waitMs: 1500 },
    { name: '03_light_curve_explorer.png', linkText: 'Light-Curve Explorer', waitMs: 4000 },
    { name: '04_detection_controls.png', linkText: 'Detection Controls', waitMs: 1500 },
    { name: '05_burst_catalog.png', linkText: 'Burst Catalog', waitMs: 2500 },
    { name: '06_event_detail.png', linkText: 'Event Detail', waitMs: 3500 },
    { name: '07_wavelet_analysis.png', linkText: 'Wavelet Analysis', waitMs: 4500 },
    { name: '08_model_training.png', linkText: 'Model Training', waitMs: 2000 },
    { name: '09_distributions.png', linkText: 'Distributions', waitMs: 3500 },
    { name: '10_export_reports.png', linkText: 'Export & Report', waitMs: 1500 },
    { name: '11_methods_reference.png', linkText: 'Methods & Help', waitMs: 1500 }
  ];

  for (const step of STEPS) {
    console.log(`[Tour] Navigating to "${step.linkText}"...`);
    try {
      // Find navigation link in sidebar
      const navLinks = await page.$$('a.nav-link');
      let clicked = false;
      for (const link of navLinks) {
        const text = await page.evaluate(el => el.textContent, link);
        if (text.includes(step.linkText)) {
          await link.click();
          clicked = true;
          break;
        }
      }

      if (!clicked) {
        console.warn(`[Tour] Link "${step.linkText}" not found, navigating directly`);
      }

      await new Promise(r => setTimeout(r, step.waitMs));

      // If on catalog, click the first event row to select it
      if (step.linkText === 'Burst Catalog') {
        const rows = await page.$$('tr[style*="cursor: pointer"], tr');
        if (rows.length > 1) {
          console.log('[Tour] Clicking first burst in catalog to select it...');
          await rows[1].click();
          await new Promise(r => setTimeout(r, 1000));
        }
      }

      const outPath = path.join(ARTIFACTS_DIR, step.name);
      await page.screenshot({ path: outPath, fullPage: false });
      console.log(`[Tour] Saved screenshot: ${step.name} (${fs.statSync(outPath).size} bytes)`);
    } catch (err) {
      console.error(`[Tour Error on ${step.linkText}]`, err.message);
    }
  }

  await browser.close();
  console.log('[Tour] Interactive walkthrough finished successfully!');
}

run().catch(err => {
  console.error('[Fatal Tour Error]', err);
  process.exit(1);
});
