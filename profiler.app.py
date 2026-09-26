# PROFILER DB PARSER BY DRAK0N
# https://github.com/antikorskiy
import queue
import os
import re
import sqlite3
import time
import threading
import configparser
from functools import lru_cache
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pystyle import Colors, Colorate, Center

# -----------------------------
# Загрузка конфигурации
# -----------------------------
CONFIG = {}

def load_config(path="profiler.conf"):
    global CONFIG
    parser = configparser.ConfigParser()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    parser.read(path)

    CONFIG["LOG_DIR"] = parser.get("PATHS", "LOG_DIR", fallback="profiler_log")
    CONFIG["DB_PATH"] = parser.get("PATHS", "DB_PATH", fallback="database/sqlite/Miras.db")
    CONFIG["UNIMPORTABLE_DIR"] = parser.get("PATHS", "UNIMPORTABLE_DIR", fallback="database/Unimportable")

    CONFIG["THREADS"] = parser.getint("GENERAL", "THREADS", fallback=12)
    CONFIG["UPDATE_INTERVAL"] = parser.getint("GENERAL", "UPDATE_INTERVAL", fallback=2)
    CONFIG["FUTURE_TIMEOUT"] = parser.getint("GENERAL", "FUTURE_TIMEOUT", fallback=60)

    CONFIG["BATCH_ROWS"] = parser.getint("SCAN_SETTINGS", "BATCH_ROWS", fallback=500)
    CONFIG["FILE_CHUNK"] = parser.getint("SCAN_SETTINGS", "FILE_CHUNK", fallback=64*1024)
    CONFIG["FAST_NORMALIZE_CACHE"] = parser.getint("SCAN_SETTINGS", "FAST_NORMALIZE_CACHE", fallback=100000)
    CONFIG["BIGRAM_THRESHOLD"] = parser.getfloat("SCAN_SETTINGS", "BIGRAM_THRESHOLD", fallback=0.34)

    CONFIG["USE_FUZZY"] = parser.getboolean("MODES", "USE_FUZZY", fallback=True)
    CONFIG["USE_FAST_NORMALIZE"] = parser.getboolean("MODES", "USE_FAST_NORMALIZE", fallback=True)
    CONFIG["CHUNKED_FILE_READ"] = parser.getboolean("MODES", "CHUNKED_FILE_READ", fallback=True)

load_config()

# -----------------------------
# Глобальные счётчики
# -----------------------------
MATCH_COUNT = 0
TOTAL_ITEMS = 0
PROGRESS_LOCK = threading.Lock()
COMPLETED = 0
CURRENT_ITEM = ""
CURRENT_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()
#STOP_EVENT = threading.Event()

LOG_DATA = {
    "start_time": None,
    "end_time": None,
    "clues": [],
    "sources": {}
}
# -----------------------------
# Баннер
# -----------------------------
def show_banner():
    return r"""
/$$$$$$$  /$$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$ /$$       /$$$$$$$$ /$$$$$$$ 
| $$__  $$| $$__  $$ /$$__  $$| $$_____/|_  $$_/| $$      | $$_____/| $$__  $$
| $$  \ $$| $$  \ $$| $$  | $$| $$        | $$  | $$      | $$      | $$  \ $$
| $$$$$$$/| $$$$$$$/| $$  | $$| $$$$$     | $$  | $$      | $$$$$   | $$$$$$$/
| $$____/ | $$__  $$| $$  | $$| $$__/     | $$  | $$      | $$__/   | $$__  $$
| $$      | $$  \ $$| $$  | $$| $$        | $$  | $$      | $$      | $$  \ $$
| $$      | $$  | $$|  $$$$$$/| $$       /$$$$$$| $$$$$$$$| $$$$$$$$| $$  | $$
|__/      |__/  |__/ \______/ |__/      |______/|________/|________/|__/  |__/
"""

banner = show_banner()
print(Colorate.Horizontal(Colors.blue_to_purple, Center.XCenter(banner)))

# -----------------------------
# Нормализация
# -----------------------------
SIMILAR_CHARS_FAST = str.maketrans({
    'a':'а','b':'в','c':'с','e':'е','o':'о','p':'р','x':'х','y':'у','k':'к',
    'A':'А','B':'В','C':'С','E':'Е','O':'О','P':'Р','X':'Х','Y':'У','K':'К',
    'м':'m','M':'М','h':'н','H':'Н','t':'т','T':'Т'
})

TRANSLIT_RU_TO_EN = str.maketrans({
    'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'e','ж':'zh','з':'z','и':'i','й':'y',
    'к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f',
    'х':'h','ц':'c','ч':'ch','ш':'sh','щ':'sch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya'
})

TRANSLIT_EN_TO_RU = str.maketrans({
    'a':'а','b':'б','c':'к','d':'д','e':'е','f':'ф','g':'г','h':'х','i':'и','j':'ж',
    'k':'к','l':'л','m':'м','n':'н','o':'о','p':'п','q':'к','r':'р','s':'с','t':'т',
    'u':'у','v':'в','w':'в','x':'кс','y':'ы','z':'з'
})

if not CONFIG["USE_FAST_NORMALIZE"]:
    fast_normalize = lambda x: x
else:
    @lru_cache(maxsize=CONFIG["FAST_NORMALIZE_CACHE"])
    def fast_normalize(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        t = text.strip().lower()
        if not t:
            return ""
        t = t.translate(SIMILAR_CHARS_FAST)
        t_en = t.translate(TRANSLIT_RU_TO_EN)
        t_ru = t.translate(TRANSLIT_EN_TO_RU)
        return f"{t} {t_en} {t_ru}"

def bigram_set(s: str):
    s = s.strip()
    if len(s) <= 1:
        return {s}
    return {s[i:i+2] for i in range(len(s)-1)}

def bigram_similarity(a: str, b: str) -> float:
    A = bigram_set(a)
    B = bigram_set(b)
    if not A and not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

# Очередь логов
log_queue = queue.Queue()

def log_writer_thread(log_file):
    while True:
        item = log_queue.get()
        if item is None:  # сигнал завершения логирования
            break
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(item + "\n")
        log_queue.task_done()

def log_write(message):
    log_queue.put(message)

# -----------------------------
# Поиск совпадений
# -----------------------------
def hard_match_fast(text: str, clues_norm: list) -> bool:
    if not text:
        return False
    text_proc = fast_normalize(text) if CONFIG["USE_FAST_NORMALIZE"] else text.lower()
    for c in clues_norm:
        if c in text_proc:
            return True
    if CONFIG["USE_FUZZY"]:
        words = re.split(r'\s+|\W+', text_proc)
        for w in words:
            if not w:
                continue
            for c in clues_norm:
                token = c.split()[0] if c else c
                if not token:
                    continue
                if abs(len(w) - len(token)) > max(1, int(len(token) * 0.6)):
                    continue
                if bigram_similarity(w, token) >= CONFIG["BIGRAM_THRESHOLD"]:
                    return True
    return False

# -----------------------------
# Логирование
# -----------------------------
if not os.path.exists(CONFIG["LOG_DIR"]):
    os.makedirs(CONFIG["LOG_DIR"])

def strip_ansi(s):
    return re.sub(r'\x1b\[[0-9;]*m', '', s)

def center_print(text):
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 80
    pad = max((term_width - len(strip_ansi(text))) // 2, 0)
    print(" " * pad + text)

def init_log(clues):
    if not os.path.exists(CONFIG["LOG_DIR"]):
        os.makedirs(CONFIG["LOG_DIR"])

    safe_clues = ",".join(clues)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(CONFIG["LOG_DIR"], f"log_profiler_{date_str}.log")

    LOG_DATA["start_time"] = datetime.now()
    LOG_DATA["clues"] = clues

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"┌[{safe_clues}]\n")
        f.write("│\n")
        f.write(f"├START: {LOG_DATA['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("├────────────────────────────\n")

    return log_file

def log_match_live(source: str, row_info: dict):
    global MATCH_COUNT
    lines = []

    header_needed = False
    with LOG_LOCK:
        if source not in LOG_DATA["sources"]:
            LOG_DATA["sources"][source] = True
            header_needed = True

    if header_needed:
        lines.extend(["│", "│", "├─┐", f"│{source}"])

    lines.append(f"│ ├─{row_info['rowid']}")
    for col, val in row_info.items():
        if col != "rowid":
            lines.append(f"│ │ ├─{col}: {val}")

    with LOG_LOCK:
        MATCH_COUNT += 1

    # Кладём все строки в очередь
    for line in lines:
        log_queue.put(line)

# -----------------------------
# Прогресс
# -----------------------------
def print_progress_dynamic(log_file, clues, start_time):
    while True:
        with PROGRESS_LOCK:
            print("\033c",end="")
            for line in banner.splitlines():
                print(Colorate.Horizontal(Colors.blue_to_purple, Center.XCenter(line)))
            print(" ")
            total = max(TOTAL_ITEMS, 1)
            percent = int((COMPLETED / total) * 100)
            bar_length = max(20, min(int(os.get_terminal_size().columns * 0.8), 100))
            filled = int(bar_length * COMPLETED / total)
            done_colored = Colorate.Horizontal(Colors.green_to_cyan, "█" * filled)
            remaining_colored = Colorate.Horizontal(Colors.red_to_purple, "█" * (bar_length - filled))
            raw_bar = f"[{done_colored}{remaining_colored}]"

            elapsed_seconds = time.time() - start_time
            speed_per_min = (COMPLETED / elapsed_seconds) * 60 if elapsed_seconds > 0 else 0
            remaining_items = total - COMPLETED
            eta_seconds = remaining_items / (COMPLETED / elapsed_seconds) if COMPLETED > 0 else 0
            eta_min = int(eta_seconds // 60)
            eta_sec = int(eta_seconds % 60)

            size_mb = 0.0
            try:
                size_mb = os.path.getsize(log_file) / (1024*1024)
            except:
                pass

            center_print(raw_bar)
            with CURRENT_LOCK:
              current = CURRENT_ITEM
            center_print(f"SCANNING: {current}")
            center_print(f"{COMPLETED}/{total}  |  {percent}%")
            center_print(f"MATCHES: {MATCH_COUNT}  |  LOG SIZE: {size_mb:.2f} MB")
            center_print(f"THREADS: {CONFIG['THREADS']}")
            center_print(f"ETA: {eta_min}m {eta_sec}s")
            center_print(f"Speed: {speed_per_min:.2f} tasks/min")
            if clues:
                center_print(f"CLUES: {', '.join(clues)}")

        if COMPLETED >= TOTAL_ITEMS:
            break
        time.sleep(CONFIG["UPDATE_INTERVAL"])

# -----------------------------
# Ввод зацепок
# -----------------------------
def get_clues():
    print(Colorate.Horizontal(Colors.cyan_to_blue, Center.XCenter("Enter clues separated by comma")))
    user_input = input("[PROFILER]>>>").strip()
    clues = [c.strip() for c in user_input.split(",") if c.strip()]
    if not clues:
        print(Colorate.Horizontal(Colors.red_to_purple, Center.XCenter("[!] No clues entered. Exiting.")))
        exit()
    return clues

# -----------------------------
# Скан таблиц SQLite
# -----------------------------
def scan_table(table, clues, clues_norm, log_file, db_path):
    global CURRENT_ITEM, MATCH_COUNT
    with CURRENT_LOCK:
        CURRENT_ITEM = f"TABLE: {table}"
    
    matches_found = 0
    try:
        conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=OFF;")
        cur.execute("PRAGMA synchronous=OFF;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA cache_size=-200000;")
        
        cur.execute(f"PRAGMA table_info('{table}')")
        cols_info = cur.fetchall()
        cols = [c[1] for c in cols_info] if cols_info else []
        if not cols:
            return 0

        offset = 0
        batch_size = CONFIG["BATCH_ROWS"]
        while True:
            col_str = ", ".join(f"[{c}]" for c in cols)
            cur.execute(f"SELECT rowid, {col_str} FROM [{table}] LIMIT ? OFFSET ?", (batch_size, offset))
            rows = cur.fetchall()
            if not rows:
                break
            offset += batch_size

            for row in rows:
                rowid = row[0]
                row_dict = {cols[i]: row[i+1] for i in range(len(cols))}
                for col, val in row_dict.items():
                    text = "" if val is None else str(val)
                    if hard_match_fast(text, clues_norm):
                        log_match_live(source=f"TABLE:{table}", row_info={"rowid": rowid, **{col: text}})
                        matches_found += 1

    except Exception as ex:
        log_write(f"[ERROR] TABLE {table} FAILED: {ex}")
    finally:
        try:
            conn.close()
        except:
            pass

    return matches_found


# -----------------------------
# Скан файлов
# -----------------------------
def scan_file(path, filename, clues, clues_norm, log_file):
    global CURRENT_ITEM, MATCH_COUNT
    with CURRENT_LOCK:
        CURRENT_ITEM = f"FILE: {filename}"

    matches_found = 0
    rowid = 0  # глобальный счетчик строк
    try:
        if CONFIG["CHUNKED_FILE_READ"]:
            with open(path, "rb") as fh:
                buffer = b""
                while True:
                    chunk = fh.read(CONFIG["FILE_CHUNK"])
                    if not chunk:
                        break
                    buffer += chunk
                    lines = buffer.split(b"\n")
                    buffer = lines.pop() if lines else b""
                    for line in lines:
                        rowid += 1
                        text = line.decode("utf-8", errors="ignore")
                        if hard_match_fast(text, clues_norm):
                            log_match_live(
                                source=f"FILE:{filename}",
                                row_info={"rowid": rowid, "line": text.strip()}
                            )
                            matches_found += 1
            # Проверка последнего остатка буфера
            if buffer:
                rowid += 1
                text = buffer.decode("utf-8", errors="ignore")
                if hard_match_fast(text, clues_norm):
                    log_match_live(
                        source=f"FILE:{filename}",
                        row_info={"rowid": rowid, "line": text.strip()}
                    )
                    matches_found += 1
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    rowid += 1
                    if hard_match_fast(line, clues_norm):
                        log_match_live(
                            source=f"FILE:{filename}",
                            row_info={"rowid": rowid, "line": line.strip()}
                        )
                        matches_found += 1
    except Exception as ex:
        log_write(f"[ERROR] Could not read {filename}: {ex}")

    return matches_found

# -----------------------------
# Финальная запись лога
# -----------------------------
def write_final_stats(log_file):
    LOG_DATA["end_time"] = datetime.now()
    duration = (LOG_DATA["end_time"] - LOG_DATA["start_time"]).total_seconds()

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("│\n│\n")
        f.write(f"├Matches: {MATCH_COUNT}\n")
        f.write(f"├Tables and files ith matches: {len(LOG_DATA['sources'])}\n")
        f.write("├Source of data:\n")
        for src in LOG_DATA["sources"]:
            f.write(f"│  {src}\n")
        f.write(f"├Scan time: {duration:.2f} сек\n")
        f.write("└End of log.\n")

# -----------------------------
# Main
# -----------------------------
def main():
    global TOTAL_ITEMS, COMPLETED

    print("\033c", end="")
    for line in banner.splitlines():
        center_print(Colorate.Horizontal(Colors.blue_to_purple, line))
    print(" ")

    clues = get_clues()
    clues_norm = [fast_normalize(c) if CONFIG["USE_FAST_NORMALIZE"] else c.lower() for c in clues]
    log_file = init_log(clues)

    # Запуск потока-логгера
    log_thread = threading.Thread(target=log_writer_thread, args=(log_file,), daemon=True)
    log_thread.start()
    start_time = time.time()

    # Определяем количество таблиц и файлов
    table_count, file_count = 0, 0
    if os.path.exists(CONFIG["DB_PATH"]):
        try:
            conn = sqlite3.connect(CONFIG["DB_PATH"],check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
            table_count = cur.fetchone()[0] or 0
            conn.close()
        except:
            table_count = 0

    if os.path.exists(CONFIG["UNIMPORTABLE_DIR"]):
        try:
            file_count = len([f for f in os.listdir(CONFIG["UNIMPORTABLE_DIR"]) if os.path.isfile(os.path.join(CONFIG["UNIMPORTABLE_DIR"], f))])
        except:
            file_count = 0

    TOTAL_ITEMS = max(1, table_count + file_count)

    # Запуск потока прогресса
    progress_thread = threading.Thread(target=print_progress_dynamic, args=(log_file, clues, start_time), daemon=True)
    progress_thread.start()

    # -----------------------------
    # Скан таблиц
    # -----------------------------
    if table_count > 0:
        try:
            conn = sqlite3.connect(CONFIG["DB_PATH"])
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [t[0] for t in cur.fetchall()]
            conn.close()
        except:
            tables = []

        with ThreadPoolExecutor(max_workers=CONFIG["THREADS"]) as exe:
            futures = {exe.submit(scan_table, t, clues, clues_norm, log_file, CONFIG["DB_PATH"]): t for t in tables}
            for future in as_completed(futures):
                try:
                    _ = future.result(timeout=CONFIG["FUTURE_TIMEOUT"])
                except Exception as ex:
                    log_write(f"[ERROR] table task failed/timeout: {ex}")
                with PROGRESS_LOCK:
                    COMPLETED += 1

    # -----------------------------
    # Скан файлов
    # -----------------------------
    if file_count > 0:
        files = [f for f in os.listdir(CONFIG["UNIMPORTABLE_DIR"]) if os.path.isfile(os.path.join(CONFIG["UNIMPORTABLE_DIR"], f))]
        with ThreadPoolExecutor(max_workers=CONFIG["THREADS"]) as exe:
            futures = {exe.submit(scan_file, os.path.join(CONFIG["UNIMPORTABLE_DIR"], f), f, clues, clues_norm, log_file): f for f in files}
            for future in as_completed(futures):
                try:
                    _ = future.result(timeout=CONFIG["FUTURE_TIMEOUT"])
                except Exception as ex:
                    log_write(f"[ERROR] file task failed/timeout: {ex}")
                with PROGRESS_LOCK:
                    COMPLETED += 1

    # Ждём завершения прогресса
    progress_thread.join()

    # Финальная запись лога
    write_final_stats(log_file)
    log_queue.put(None)  # сигнал завершения логгера
    log_thread.join()    # ждём пока поток закончит
    # Вывод DONE
    center_print(Colorate.Horizontal(
        Colors.green_to_cyan,
        f"\n=== DONE ===\n"
        f"LOG SAVED: {os.path.abspath(log_file)}\n"
        f"MATCHES: {MATCH_COUNT}"
    ))
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        center_print(Colorate.Horizontal(Colors.red_to_purple, "\n[!] PROFILER INTERRUPTED BY USER"))
    except Exception as e:
        center_print(Colorate.Horizontal(Colors.red_to_purple, f"\n[ERROR] Unexpected error: {e}"))
        