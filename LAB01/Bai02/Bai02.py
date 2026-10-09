
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from datetime import datetime, date


BASE_URL = "https://lichsugia.site/lich-su-giao-dich/FPT.html"

START_DATE = date(2021, 10, 9)
END_DATE = date(2026, 10, 9)

MAX_PAGES = 40
DELAY = 2

session = requests.Session()
session.headers.update({
    "User-Agent": "FPT-Student-Research/1.0"
})




def parse_price(value):
    value = value.strip().replace(" ", "")
    value = value.replace(",", ".")
    return float(value)


def parse_volume(value):
    value = re.sub(r"[^\d]", "", value)
    return int(value)


def parse_percent(value):
    match = re.search(
        r"\(([+-]?\d+(?:[.,]\d+)?)\s*%\)",
        value
    )

    if match:
        return float(match.group(1).replace(",", "."))

    return None


def crawl_page(page):

    url = f"{BASE_URL}?page={page}"

    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()

    except requests.RequestException as e:
        print("Loi truy cap:", e)
        return None

    soup = BeautifulSoup(response.text, "html.parser")


    table = None

    for t in soup.find_all("table"):
        headings = [
            th.get_text(" ", strip=True)
            for th in t.find_all("th")
        ]

        if (
            "Ngày" in headings
            and "Khối lượng" in headings
            and "Mở cửa" in headings
        ):
            table = t
            break

    if table is None:
        print("Khong tim thay bang tai trang:", page)
        return None

    data = []

    for row in table.find_all("tr"):

        cells = row.find_all("td")

        if len(cells) != 9:
            continue

        values = [
            cell.get_text(" ", strip=True)
            for cell in cells
        ]

        try:
            trading_date = datetime.strptime(
                values[0], "%d/%m/%Y"
            ).date()

            record = {
                "symbol": "FPT",
                "date": trading_date,
                "open": parse_price(values[6]),
                "high": parse_price(values[7]),
                "low": parse_price(values[8]),
                "close": parse_price(values[2]),
                "adjusted_close": parse_price(values[1]),
                "volume": parse_volume(values[4]),
                "change_percent": parse_percent(values[3])
            }

            data.append(record)

        except (ValueError, IndexError):
            continue

    return data


all_data = []
reached_start = False
seen_pages = set()

for page in range(1, MAX_PAGES + 1):

    print(f"Dang thu thap trang {page}...")

    page_data = crawl_page(page)

    if page_data is None:
        print("Loi truy cap.")
        break

    if len(page_data) == 0:
        print("Khong con du lieu.")
        break

    page_signature = (
        max(item["date"] for item in page_data),
        min(item["date"] for item in page_data)
    )

    if page_signature in seen_pages:
        print("Trang du lieu bi lap.")
        break

    seen_pages.add(page_signature)

    oldest_date = min(item["date"] for item in page_data)

    # Chi giu du lieu trong khoang 5 nam
    valid_data = [
        item for item in page_data
        if START_DATE <= item["date"] <= END_DATE
    ]

    all_data.extend(valid_data)

    print("So ban ghi hop le:", len(valid_data))
    print("Ngay cu nhat trang:", oldest_date)

    # Dung khi da den moc 5 nam
    if oldest_date <= START_DATE:
        reached_start = True
        print("Da den moc 5 nam.")
        break

    time.sleep(DELAY)

    
df = pd.DataFrame(all_data)

if not df.empty:

    df = df.drop_duplicates(
        subset=["symbol", "date"]
    )

    df = df.sort_values("date").reset_index(drop=True)

    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")



    df.to_csv(
        "FPT_stock.csv",
        index=False,
        encoding="utf-8-sig"
    )

    df.to_csv(
        "FPT_stock.tsv",
        sep="\t",
        index=False
    )

    df.to_json(
        "FPT_stock.json",
        orient="records",
        force_ascii=False,
        indent=4
    )

    # =========================================
    # HIEN THI KET QUA
    # =========================================

    print("\n===== KET QUA THU THAP =====")

    print("Ma chung khoan: FPT")
    print("Tong so ban ghi:", len(df))
    print("Ngay bat dau:", df["date"].min())
    print("Ngay ket thuc:", df["date"].max())

    if not reached_start:
        print("Chua xac nhan du 5 nam du lieu.")

    print("\nThong ke du lieu:")
    print(df.describe())

    print("\n10 dong dau tien:")
    display(df.head(10))

    print("\n10 dong cuoi cung:")
    display(df.tail(10))

    print("\nHoan thanh.")

else:
    print("Khong thu thap duoc du lieu.")
