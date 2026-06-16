const puppeteer = require("puppeteer");
const fs = require("fs");
const path = require("path");
const axios = require("axios");
const CAR_YEAR_Ar = [
  "p11-1996-1999",
  "p11-restajling-1999-2002",
  "p12-2001-2012",
];
for (let i = 0; i < CAR_YEAR_Ar.length; i++) {
  (async () => {
    const MAX_ADS_TO_PROCESS = 50;
    const CAR_BRAND = "nissan";
    const CAR_MODEL = "primera";

    const CATALOG_URL = `https://cars.av.by/${CAR_BRAND}/${CAR_MODEL}/${CAR_YEAR_Ar[i]}`;
    const SAVE_ROOT = path.resolve(
      __dirname,
      `downloaded_images/${CAR_BRAND}/${CAR_MODEL}/${CAR_BRAND}_${CAR_MODEL}-${CAR_YEAR_Ar[i]}`,
    );
    fs.mkdirSync(SAVE_ROOT, { recursive: true });

    console.log("Starting browser...");
    const browser = await puppeteer.launch({
      headless: "new",
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });

    const page = await browser.newPage();
    const UA =
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36";
    await page.setUserAgent(UA);
    await page.setViewport({ width: 1920, height: 1080 });

    console.log(`Navigating to the initial catalog page: ${CATALOG_URL}`);
    await page.goto(CATALOG_URL, { waitUntil: "domcontentloaded" });

    try {
      console.log("Checking for cookie consent banner...");
      const cookieButtonSelector = "div.cookie-banner button.button--default";
      await page.waitForSelector(cookieButtonSelector, { timeout: 5000 });
      await page.click(cookieButtonSelector);
      console.log("Cookie consent banner accepted.");
      await page.waitForTimeout(1000);
    } catch (e) {
      console.log("Cookie consent banner not found or already accepted.");
    }

    const showMoreButtonSelector = ".paging__button a";
    const adItemSelector = "listing-item";
    let clickCount = 0;
    while (clickCount < 20) {
      try {
        console.log(
          `\n[Итерация ${clickCount + 1}] Скроллим для активации кнопки...`,
        );
        await page.evaluate(() => window.scrollBy(0, window.innerHeight * 2));
        await new Promise((r) => setTimeout(r, 2000));
        const isButtonExist = await page.$(showMoreButtonSelector);
        if (!isButtonExist) {
          console.log(
            "Кнопка 'Показать ещё' отсутствует. Похоже, это все объявления.",
          );
          break;
        }

        const adCountBefore = await page.$$eval(
          adItemSelector,
          (nodes) => nodes.length,
        );
        console.log(`Объявлений в DOM: ${adCountBefore}. Кликаем...`);

        await page.evaluate((selector) => {
          const btn = document.querySelector(selector);
          if (btn) btn.click();
        }, showMoreButtonSelector);

        await page.waitForFunction(
          (selector, prevCount) =>
            document.querySelectorAll(selector).length > prevCount,
          { timeout: 10000 },
          adItemSelector,
          adCountBefore,
        );

        const adCountAfter = await page.$$eval(
          adItemSelector,
          (nodes) => nodes.length,
        );
        console.log(`Успешно! Теперь объявлений: ${adCountAfter}`);

        clickCount++;
      } catch (error) {
        console.error(`Остановка цикла: ${error.message}`);
        break;
      }
    }

    console.log("\nFinished loading all ads. Scraping all links...");
    const allAdLinks = await page.$$eval(
      "div.listing__items div.listing-item__wrap a.listing-item__link",
      (nodes) => nodes.map((n) => n.href),
    );

    const uniqueLinks = [...new Set(allAdLinks)];
    console.log(`Found ${uniqueLinks.length} unique ads in total.`);
    for (let i = 0; i < Math.min(uniqueLinks.length, MAX_ADS_TO_PROCESS); i++) {
      const link = uniqueLinks[i];
      console.log(
        `\n[${i + 1}/${Math.min(uniqueLinks.length, MAX_ADS_TO_PROCESS)}] Processing: ${link}`,
      );

      try {
        await page.goto(link, { waitUntil: "domcontentloaded" });

        try {
          await page.waitForSelector("div.gallery__stage", { timeout: 7000 });
          await page.click("div.gallery__stage");
          await page.waitForSelector("div.fullscreen-gallery--active", {
            visible: true,
            timeout: 5000,
          });
        } catch (e) {}

        const imageUrls = await page.evaluate(() => {
          const urls = new Set();
          let images = document.querySelectorAll(
            "div.fullscreen-gallery--active div.fullscreen-gallery__item img",
          );
          if (images.length === 0) {
            images = document.querySelectorAll(
              "div.gallery__thumbs-frame a.gallery__thumb",
            );
            images.forEach((thumb) => {
              if (thumb && thumb.href) urls.add(thumb.href);
            });
          } else {
            images.forEach((img) => {
              const imageUrl = img.getAttribute("data-src") || img.src;
              if (imageUrl) urls.add(new URL(imageUrl, document.baseURI).href);
            });
          }
          return Array.from(urls);
        });

        console.log(`Found ${imageUrls.length} unique image URLs.`);
        if (imageUrls.length === 0) continue;

        let savedCount = 0;
        for (let j = 0; j < imageUrls.length; j++) {
          try {
            const response = await axios.get(imageUrls[j], {
              responseType: "arraybuffer",
              timeout: 20000,
              headers: { "User-Agent": UA, Referer: link },
            });
            if (response.status === 200 && response.data.length > 10000) {
              const filename = path.basename(new URL(imageUrls[j]).pathname);
              fs.writeFileSync(
                path.join(SAVE_ROOT, `ad_${i + 1}_${filename}`),
                response.data,
              );
              savedCount++;
            }
          } catch (err) {}
        }
        console.log(
          `Successfully saved ${savedCount} of ${imageUrls.length} images.`,
        );
      } catch (pageError) {
        console.error(`Failed to process page ${link}: ${pageError.message}`);
      }
    }

    await browser.close();
    console.log("\nScraping complete.");

    async function autoScroll(page) {}
  })();
}
