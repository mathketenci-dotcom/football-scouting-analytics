# Football Scouting Analytics

Maç görüntülerinden (video) ve/veya hazır tracking verisinden yola çıkarak
**pitch control** ve **"en iyi hamle" (best decision) analizi** yapmayı
hedefleyen bir scouting/analitik projesi.

## Amaç

Futbolda oyuncu ve top konumlarının her karede (frame) bilinmesi, bir aksiyon
anında sahadaki tüm oyuncuların hangi bölgeyi kontrol ettiğini (pitch control)
hesaplamaya imkan verir. Bu proje şunları amaçlar:

- Belirli bir anda hangi takımın hangi bölgeyi kontrol ettiğini modellemek,
- Topa sahip oyuncunun mevcut alternatifleri (pas, sürüş, şut) arasından
  hangisinin en yüksek beklenen değeri (Expected Possession Value / EPV)
  ürettiğini, yani **o an için "en iyi hamlenin" ne olduğunu** tespit etmek,
- Bu analizleri scouting amacıyla oyuncu/takım performansını değerlendirmek
  için kullanmak.

Pitch control ve EPV modelleri için Laurie Shaw'ın (@EightyFivePoint)
`LaurieOnTracking` kütüphanesindeki referans implementasyonları temel alınır.

## İki Veri Kaynağı Yaklaşımı

Proje iki paralel veri kaynağıyla ilerliyor:

1. **Hazır tracking verisi (Metrica sample-data):** `Metrica_IO` ile
   doğrudan okunabilen, oyuncu/top x-y koordinatlarını içeren CSV verisi.
   Pitch control ve EPV modellerini hızlıca doğrulamak/prototiplemek için
   kullanılır.
2. **Video tabanlı yaklaşım (computer vision):** Gerçek maç görüntülerinden
   kare kare (frame-by-frame) görüntü çıkarılıp, ileride bir computer vision
   modeliyle (oyuncu/top tespiti ve takibi) bu görüntülerden tracking verisi
   üretilmesi hedeflenir. Böylece pitch control / EPV analizi, hazır tracking
   verisi olmayan herhangi bir maç görüntüsüne de uygulanabilir hale gelir.

## Proje Yapısı

```
football-scouting-analytics/
├── LaurieOnTracking/       # Metrica_IO, Metrica_Velocities, Metrica_PitchControl,
│                           # Metrica_EPV, Metrica_Viz modülleri ve tutorial'lar
├── sample-data/            # Metrica Sports örnek maç verileri (Sample_Game_1/2/3)
│   └── data/               # (yerel, .gitignore'da — repoya push edilmez)
├── video_raw/               # İndirilen ham video, kesilmiş klipler ve çıkarılan
│                            # kareler (yerel, .gitignore'da — repoya push edilmez;
│                            # telif hakkı nedeniyle kaynak video asla paylaşılmaz)
├── test_load_tracking.py   # Home/Away tracking + top verisini yükleyip dogrulayan test scripti
├── requirements.txt
└── README.md
```

**Önemli:** `sample-data/` ve `video_raw/` klasörleri `.gitignore` ile hariç
tutulmuştur. Ham tracking verisi boyut nedeniyle, ham video ve kareler ise
telif hakkı nedeniyle repoya asla push edilmez; sadece yerelde tutulur.

## Veri

### Tracking verisi

`sample-data/data/Sample_Game_1` altında:

- `Sample_Game_1_RawTrackingData_Home_Team.csv` — ev sahibi takımın oyuncu
  konumları (her satır bir frame; her oyuncu için x/y kolonları)
- `Sample_Game_1_RawTrackingData_Away_Team.csv` — deplasman takımının oyuncu
  konumları
- `Sample_Game_1_RawEventsData.csv` — pas, şut, top kaybı gibi olay (event)
  kayıtları

**Not:** Top (ball) pozisyonu ayrı bir dosyada gelmez; her tracking
dosyasının son iki kolonu (`ball_x`, `ball_y`) top konumunu içerir.

### Video verisi

`video_raw/` altında (yerel, gitignore'da):

- `source.mp4` — indirilen kaynak video (ses + görüntü birleştirilmiş)
- `clip_01.mp4` — kaynaktan kesilen 9 saniyelik örnek klip (0:23–0:32)
- `frames_clip_01/` — `clip_01.mp4`'ten kare kare çıkarılmış PNG görüntüler
  (25 fps × 9 sn = 225 kare)

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Video işleme için ayrıca [yt-dlp](https://github.com/yt-dlp/yt-dlp) (video
indirme) ve [ffmpeg](https://ffmpeg.org/) (kesme + kare çıkarma) gereklidir.

## Kullanım

### Tracking verisini doğrulama

```bash
python test_load_tracking.py
```

Bu script `LaurieOnTracking/Metrica_IO.py` içindeki `tracking_data` ve
`read_event_data` fonksiyonlarını kullanarak Sample_Game_1 için Home/Away
tracking verisini ve event verisini yükler, temel bütünlük kontrollerini
(boş olmama, top kolonlarının varlığı) yapar.

### Video indirme, kesme ve kare çıkarma

```bash
# Video indir (video + ses ayrı akış olarak gelebilir, ffmpeg ile birleştirilir)
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b" -o "video_raw/source.%(ext)s" <video_url>

# 0:23-0:32 arasını kes
ffmpeg -i video_raw/source.mp4 -ss 00:00:23 -to 00:00:32 \
  -c:v libx264 -crf 18 -c:a aac video_raw/clip_01.mp4

# Kareleri tek tek PNG olarak çıkar
ffmpeg -i video_raw/clip_01.mp4 video_raw/frames_clip_01/frame_%04d.png
```

## Sonraki Adımlar

- `Metrica_Velocities.py` ile oyuncu hız/ivme hesaplaması
- `Metrica_PitchControl.py` ile pitch control modeli
- `Metrica_EPV.py` ile Expected Possession Value ve "en iyi hamle" analizi
- Bulguların `Metrica_Viz.py` ile görselleştirilmesi
- Video karelerinden (computer vision ile) oyuncu/top tespiti ve takibi
  yapılarak, gerçek maç görüntülerinden tracking verisi üretilmesi ve bu
  verinin pitch control / EPV modellerine girdi olarak kullanılması

## Kaynaklar

- [Metrica Sports sample data](https://github.com/metrica-sports/sample-data)
- [Laurie Shaw — LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [ffmpeg](https://ffmpeg.org/)
