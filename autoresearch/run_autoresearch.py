import ast
import os
import sys
import time
import json
import re
import requests
import subprocess
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import shutil

# Load environment variables (e.g., OPENROUTER_API_KEY)
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not found in environment or .env file.")
    print("Please add it to the .env file.")

# Primary and Fallback Models (via OpenRouter)
PRIMARY_MODEL = "stepfun/step-3.5-flash:free"
FALLBACK_MODEL = "google/gemini-2.5-flash-lite"
# Direct Gemini API fallback model
GEMINI_DIRECT_MODEL = "gemini-2.0-flash"

LOG_FILE = "experiment_logs.csv"

def query_gemini_direct(prompt, system_prompt=""):
    """Call Google Gemini API directly as a last-resort fallback."""
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not set. Cannot use direct Gemini fallback.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_DIRECT_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Error querying Gemini direct API: {e}")
        return None

def query_openrouter(prompt, system_prompt="", use_fallback=False):
    model = FALLBACK_MODEL if use_fallback else PRIMARY_MODEL

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/karpathy/autoresearch",
        "X-Title": "Autoresearch Quant Agent"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error querying {model}: {e}")
        if not use_fallback:
            print(f"Attempting fallback to {FALLBACK_MODEL} via OpenRouter...")
            return query_openrouter(prompt, system_prompt, use_fallback=True)
        else:
            print("OpenRouter fallback also failed. Trying direct Gemini API...")
            return query_gemini_direct(prompt, system_prompt)

def extract_code(text):
    """Extract python code from markdown code blocks."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1)
    
    # If no python tag, try generic code block
    match = re.search(r"```\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1)
        
    return text  # Assume the whole response is code if no blocks found

def run_experiment(iteration_num):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting iteration {iteration_num}")
    
    # 1. Read the current context
    with open("program.md", "r") as f:
         program_rules = f.read()
         
    with open("train.py", "r") as f:
         current_train_code = f.read()
         
    # Read history to give context — last 20 runs so the LLM doesn't repeat failed approaches
    history = ""
    if os.path.exists(LOG_FILE):
        logs_df = pd.read_csv(LOG_FILE)
        history_df = logs_df.tail(20)
        history = "\nRecent Experiment History (last 20 runs):\n" + history_df.to_string()
        # Also surface the single best run so the LLM knows the current champion
        valid_logs = logs_df[(logs_df['Status'] == 'Completed') & (logs_df['val_max_drawdown'] <= 0.15)]
        if not valid_logs.empty:
            best_row = valid_logs.loc[valid_logs['val_excess_return'].idxmax()]
            history += f"\n\nCurrent Best Run (champion to beat):\n{best_row.to_string()}"
         
    # 2. Formulate Prompt
    system_prompt = "You are an expert Quantitative Machine Learning Researcher."
    prompt = f"""
    Here are the overarching rules for your algorithmic trading portfolio optimizations:
    {program_rules}
    
    {history}
    
    Here is the current code for `train.py`:
    ```python
    {current_train_code}
    ```
    
    Propose exactly ONE targeted modification to improve `val_excess_return` while keeping `val_max_drawdown` above -0.15 (15% loss).
    Return the ENTIRE completely valid python script for `train.py` inside a single ```python codeblock.
    Do not omit any code or use comments like "# rest of code here". Write the full modified file.
    Before the codeblock, briefly explain your hypothesis.
    """
    
    # 3. Get LLM Proposal
    print("Querying LLM for new hypothesis...")
    llm_response = query_openrouter(prompt, system_prompt)
    
    if not llm_response:
        print("Failed to get response from LLMs. Skipping iteration.")
        return False
        
    new_code = extract_code(llm_response)

    # Validate syntax before touching train.py
    try:
        ast.parse(new_code)
    except SyntaxError as e:
        print(f"Syntax error in generated code: {e}. Skipping iteration.")
        return False

    # Backup current best train.py
    shutil.copyfile("train.py", "train.py.best")

    # Write new experiment
    with open("train.py", "w") as f:
        f.write(new_code)
        
    # 4. Run the backtest
    print("Running training simulation (May take up to 5 mins)...")
    try:
        # Run with timeout matching our internal py timeout (+ buffer)
        result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True, timeout=330)
        output = result.stdout
        errors = result.stderr
        
        # 5. Parse KPIs
        val_excess_return = None
        val_max_drawdown = None
        
        # Look for [FINAL RESULTS] block outputs defined in train.py
        excess_match = re.search(r"val_excess_return=([0-9.\-]+)", output)
        if excess_match: val_excess_return = float(excess_match.group(1))
            
        drawdown_match = re.search(r"val_max_drawdown=([0-9.\-]+)", output)
        if drawdown_match: val_max_drawdown = float(drawdown_match.group(1))
        
        status = "Completed" if (val_excess_return is not None and val_max_drawdown is not None) else "Failed_Parse"
        if result.returncode != 0:
            status = "Failed_Run"
            print(f"Error during runtime:\n{errors}")
            
    except subprocess.TimeoutExpired:
        status = "Timeout"
        output = "Process exceeded 5 minute + buffer timeout."
        val_excess_return, val_max_drawdown = None, None
        
    # 6. Logging
    log_entry = {
         "Iteration": iteration_num,
         "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
         "Status": status,
         "val_excess_return": val_excess_return,
         "val_max_drawdown": val_max_drawdown,
         "Hypothesis_Summary": llm_response.split("```")[0].strip()[:200].replace('\n', ' ') # Save first 200 chars of thought process
    }
    
    df = pd.DataFrame([log_entry])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)
        
    print(f"Iteration Results: Status={status} | Excess Return={val_excess_return} | Drawdown={val_max_drawdown}")
    
    # 7. Rollback or Keep (Greedy Search)
    # We only keep if the run succeeded, AND improved excess return, AND drawdown is acceptable (< 15%)
    # Let's read the best previous log entry
    best_excess = float('-inf')
    if os.path.exists(LOG_FILE):
        logs = pd.read_csv(LOG_FILE)
        valid_logs = logs[(logs['Status'] == 'Completed') & (logs['val_max_drawdown'] <= 0.15)]
        if not valid_logs.empty:
            best_excess = valid_logs['val_excess_return'].max()
            
    if status == "Completed" and val_excess_return > best_excess and val_max_drawdown <= 0.15:
         print("🌟 NEW BEST MODEL FOUND! Accepting changes and archiving! 🌟")
         # Archive the winning model and script
         archive_dir = f"saved_models/iteration_{iteration_num}"
         os.makedirs(archive_dir, exist_ok=True)
         shutil.copyfile("train.py", os.path.join(archive_dir, "train.py"))
         if os.path.exists("best_model.pt"):
             shutil.copyfile("best_model.pt", os.path.join(archive_dir, "best_model.pt"))
    else:
         print("Hypothesis rejected. Reverting train.py to previous best state.")
         shutil.copyfile("train.py.best", "train.py")
         
    # Optional Discord Webhook Notification
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    if DISCORD_WEBHOOK_URL:
        emoji = "🚀" if status == "Completed" and val_excess_return > best_excess and val_max_drawdown <= 0.15 else "❌"
        msg = f"{emoji} **Autoresearch Iteration {iteration_num}**\n**Status:** {status}\n**Alpha (Excess Return):** {val_excess_return}\n**Max Drawdown:** {val_max_drawdown}\n**Hypothesis:** {log_entry['Hypothesis_Summary']}"
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=10)
        except Exception as e:
            print(f"Failed to trigger Discord webhook: {e}")
            
    return True

def main():
    print("Starting Autonomous Quant Researcher Loop...")
    print(f"Primary Model: {PRIMARY_MODEL} | Fallback: {FALLBACK_MODEL}")

    iter_num = 1
    # Check what iteration we are on
    if os.path.exists(LOG_FILE):
        logs = pd.read_csv(LOG_FILE)
        if not logs.empty:
            iter_num = logs['Iteration'].max() + 1

    # Wall-clock budget: exit after 50 min so the CI job has time to commit before GitHub's 6h limit
    CI_TIME_BUDGET = 50 * 60
    loop_start = time.time()

    while True:
        try:
            run_experiment(iter_num)
            iter_num += 1
            elapsed = time.time() - loop_start
            if elapsed >= CI_TIME_BUDGET:
                print(f"CI time budget reached ({elapsed/60:.1f} min). Exiting to commit results.")
                break
            print("\nWaiting 10 seconds before next iteration...")
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nAutonomous Loop Stopped by User.")
            break

if __name__ == '__main__':
    main()
