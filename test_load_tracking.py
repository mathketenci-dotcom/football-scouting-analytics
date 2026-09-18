"""
Metrica_IO kullanarak Sample_Game_1 icin Home/Away tracking verisini yukler
ve dogrular. Top (ball) pozisyonu ayri bir dosya degil; Metrica formatinda
her tracking dosyasinin son iki kolonu (ball_x, ball_y) olarak gelir, bu
yuzden home/away tracking verisinin icinde kontrol edilir.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "LaurieOnTracking"))

import Metrica_IO as mio

DATADIR = str(ROOT / "sample-data" / "data")
GAME_ID = 1


def main():
    print(f"Veri klasoru: {DATADIR}")

    tracking_home = mio.tracking_data(DATADIR, GAME_ID, "Home")
    tracking_away = mio.tracking_data(DATADIR, GAME_ID, "Away")
    events = mio.read_event_data(DATADIR, GAME_ID)

    assert not tracking_home.empty, "Home tracking verisi bos geldi"
    assert not tracking_away.empty, "Away tracking verisi bos geldi"
    assert not events.empty, "Event verisi bos geldi"
    assert "ball_x" in tracking_home.columns and "ball_y" in tracking_home.columns, \
        "Top (ball) pozisyon kolonlari tracking verisinde bulunamadi"

    print(f"Home tracking: {tracking_home.shape[0]} satir, {tracking_home.shape[1]} kolon")
    print(f"Away tracking: {tracking_away.shape[0]} satir, {tracking_away.shape[1]} kolon")
    print(f"Event verisi:  {events.shape[0]} satir, {events.shape[1]} kolon")

    print("\nHome tracking ilk 3 satir (ozet kolonlar):")
    preview_cols = [c for c in tracking_home.columns[:4]] + ["ball_x", "ball_y"]
    print(tracking_home[preview_cols].head(3))

    print("\nTum kontroller basarili: Home/Away tracking ve top verisi dogru sekilde yuklendi.")


if __name__ == "__main__":
    main()
