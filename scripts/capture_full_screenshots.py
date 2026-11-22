#!/usr/bin/env python3
"""Run Streamlit and capture beautiful, full-height screenshots of each tab."""
import os
import subprocess
import time
from pathlib import Path
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = str(ROOT / "figures")
PORT = 8501
URL = f"http://localhost:{PORT}"

# 1. Start Streamlit server
print("Starting Streamlit server...")
proc = subprocess.Popen(
    ["python3", "-m", "streamlit", "run", "src/app.py",
     "--server.port", str(PORT),
     "--server.headless", "true",
     "--browser.gatherUsageStats", "false"],
    cwd=str(ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

try:
    # Wait for server to become responsive
    ready = False
    for i in range(40):
        try:
            r = requests.get(URL, timeout=2)
            if r.status_code == 200:
                ready = True
                print("Streamlit server is ready!")
                break
        except Exception:
            pass
        time.sleep(1)

    if not ready:
        raise RuntimeError("Streamlit server failed to start within 40 seconds.")

    # 2. Setup Chrome
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    # 1350 wide, 1150 tall to capture both header and complete plots without cutoffs
    opts.add_argument("--window-size=1350,1150")
    opts.add_argument("--force-device-scale-factor=1.5")

    driver = webdriver.Chrome(options=opts)
    driver.get(URL)

    # Wait for data fetching & rendering
    print("Waiting for data to load in browser (up to 90s)...")
    wait = WebDriverWait(driver, 90)
    # Wait until the tabs or metrics appear
    wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(., 'Surface') or @data-baseweb='tab']")))
    time.sleep(3)
    print("Data loaded successfully!")

    # Screenshot 1: Surface tab
    driver.save_screenshot(f"{OUTDIR}/dashboard_surface.png")
    print("✓ Saved dashboard_surface.png")

    # Locate tabs
    tab_buttons = driver.find_elements(By.XPATH, "//button[@role='tab' or @data-baseweb='tab']")
    print(f"Found {len(tab_buttons)} tab buttons: {[t.text for t in tab_buttons]}")

    # Screenshot 2: Deeper layers tab
    if len(tab_buttons) >= 2:
        tab_buttons[1].click()
        time.sleep(4)
        driver.save_screenshot(f"{OUTDIR}/dashboard_deep.png")
        print("✓ Saved dashboard_deep.png")

    # Screenshot 3: Evaluation tab
    if len(tab_buttons) >= 3:
        tab_buttons[2].click()
        time.sleep(4)
        driver.save_screenshot(f"{OUTDIR}/dashboard_eval.png")
        print("✓ Saved dashboard_eval.png")

    driver.quit()
    print("Screenshots captured successfully!")

finally:
    proc.terminate()
    proc.wait()
    print("Streamlit server stopped.")
