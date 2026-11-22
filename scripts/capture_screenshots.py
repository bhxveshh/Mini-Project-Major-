#!/usr/bin/env python3
"""Capture screenshots of the Streamlit dashboard using Selenium."""
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = str(ROOT / "figures")
URL = "http://localhost:8501"

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--disable-gpu")
opts.add_argument("--window-size=1400,900")
opts.add_argument("--force-device-scale-factor=1.5")

driver = webdriver.Chrome(options=opts)
driver.get(URL)

# Wait for the page to fully load (app needs to fetch API data)
print("Waiting for Streamlit app to load...")
time.sleep(25)

# Check page content
body = driver.find_element(By.TAG_NAME, "body").text
if "Could not build" in body:
    print(f"App error detected. Page text:\n{body[:500]}")
    # Still capture for reference
    driver.save_screenshot(f"{OUTDIR}/dashboard_surface.png")
    print("✗ App has an error state")
    driver.quit()
    exit(1)

# Screenshot 1: Full page - Surface tab (default)
driver.save_screenshot(f"{OUTDIR}/dashboard_surface.png")
print("✓ Captured dashboard_surface.png")

# Find tabs using different selectors
tab_buttons = driver.find_elements(By.CSS_SELECTOR, 'button[data-baseweb="tab"]')
if not tab_buttons:
    tab_buttons = driver.find_elements(By.CSS_SELECTOR, '[role="tab"]')
if not tab_buttons:
    tab_buttons = driver.find_elements(By.CSS_SELECTOR, '.stTabs button')
    
print(f"Found {len(tab_buttons)} tab buttons")
for i, t in enumerate(tab_buttons):
    print(f"  Tab {i}: '{t.text}'")

# Screenshot 2: Deeper layers tab
if len(tab_buttons) >= 2:
    try:
        tab_buttons[1].click()
        time.sleep(5)
        driver.save_screenshot(f"{OUTDIR}/dashboard_deep.png")
        print("✓ Captured dashboard_deep.png")
    except Exception as e:
        print(f"✗ Error clicking deep tab: {e}")

# Screenshot 3: Evaluation tab
if len(tab_buttons) >= 3:
    try:
        tab_buttons[2].click()
        time.sleep(4)
        driver.save_screenshot(f"{OUTDIR}/dashboard_eval.png")
        print("✓ Captured dashboard_eval.png")
    except Exception as e:
        print(f"✗ Error clicking eval tab: {e}")

driver.quit()
print("\nDone!")
