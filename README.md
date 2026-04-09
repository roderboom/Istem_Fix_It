# 🔧 FixBot — עוזר תיקונים חכם לטלגרם

בוט טלגרם שמנתח תמונות של דברים שבורים ומחזיר הוראות תיקון מפורטות עם סימונים על התמונה.
**עובד לחלוטין מקומית** — ללא שירותי ענן, ללא עלות, ללא שליחת מידע לאינטרנט.

---

## 🏗️ ארכיטקטורה

```
משתמש (עברית)
     │
     ▼
 [Telegram Bot]
     │
     ├─► translator.py  →  עברית ➜ אנגלית  (קלט)
     │
     ├─► vision.py      →  Ollama API (LLaMA 3.2-Vision 11B)
     │
     ├─► translator.py  →  אנגלית ➜ עברית  (פלט)
     │
     └─► annotator.py   →  ציור סימונים על התמונה (Pillow)
          │
          ▼
     משתמש מקבל תמונה מסומנת + הוראות בעברית
```

שני קונטיינרים ב-Podman:
- fixbot-ollama  שרת מודל הבינה המלאכותית (GPU)
- fixbot-bot     בוט הטלגרם (Python)

---

## דרישות מערכת

| רכיב | דרישה מינימלית | המחשב שלך |
|------|----------------|-----------|
| GPU | 8 GB VRAM | RTX 5070 Ti 16 GB |
| RAM | 16 GB | 94 GB |
| מערכת הפעלה | Linux | |
| מנהל התקן NVIDIA | 520+ | |

---

## שלב 1 — התקנת Podman

```bash
sudo apt-get update
sudo apt-get install -y podman
```

בדיקה:
```bash
podman --version
```

---

## שלב 2 — התקנת podman-compose

```bash
pip install podman-compose
```

בדיקה:
```bash
podman-compose --version
```

---

## שלב 3 — התקנת NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

בדיקה:
```bash
nvidia-ctk --version
nvidia-smi
```

---

## שלב 4 — הגדרת CDI (גישת GPU לקונטיינרים)

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  docker.io/nvidia/cuda:12.3.0-base-ubuntu22.04 \
  nvidia-smi -L
```

אם מציג "GPU 0: NVIDIA GeForce RTX 5070 Ti" — הכל תקין.

---

## שלב 5 — קבלת טוקן טלגרם

1. פתחו טלגרם וחפשו @BotFather
2. שלחו /newbot
3. בחרו שם לבוט
4. בחרו שם משתמש (חייב להסתיים ב-bot)
5. שמרו את הטוקן — נראה כך: 7123456789:AAFxxxxx...

---

## שלב 6 — הגדרת הפרויקט

```bash
tar xzf fixbot.tar.gz
cd fixbot
cp .env.example .env
nano .env
```

שנו את השורה:
  TELEGRAM_TOKEN=your_telegram_bot_token_here
ל:
  TELEGRAM_TOKEN=הטוקן-שלכם-כאן

שמרו: Ctrl+O → Enter → Ctrl+X

---

## שלב 7 — בניית הקונטיינרים

```bash
podman-compose build
```

הבנייה לוקחת 10-20 דקות בפעם הראשונה.
מורידה: Python, ספריות, גופנים עבריים, מודלי תרגום (~200 MB).

---

## שלב 8 — הורדת מודל הבינה המלאכותית

```bash
podman-compose up -d ollama
sleep 15
podman exec fixbot-ollama ollama pull llama3.2-vision:11b
```

ההורדה: ~8 GB, לוקחת 5-15 דקות.

---

## שלב 9 — הפעלת הבוט

```bash
podman-compose up -d fixbot
```

בדיקה:
```bash
podman-compose logs -f
```

אמורים להופיע:
  fixbot-bot | INFO | FixBot is running...

---

## שלב 10 — בדיקה בטלגרם

1. פתחו טלגרם
2. חפשו את הבוט לפי שם המשתמש שבחרתם
3. שלחו /start
4. שלחו תמונה של משהו שבור עם כיתוב בעברית

---

## פקודות הבוט

/start  הודעת ברוכים הבאים
/help   טיפים לצילום טוב
/reset  מחיקת השיחה והתחלה מחדש

---

## פקודות שימושיות

```bash
# לוגים בזמן אמת
podman-compose logs -f

# עצירת הכל
podman-compose down

# הפעלה מחדש של הבוט אחרי שינויי קוד
podman-compose build fixbot && podman-compose up -d fixbot

# בדיקת Ollama
curl http://localhost:11434/api/tags
```

---

## פתרון בעיות

Cannot connect to Ollama:
  podman ps | grep ollama
  podman-compose logs ollama

GPU לא נגיש:
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

מודל לא נמצא:
  podman exec fixbot-ollama ollama pull llama3.2-vision:11b

שגיאת בנייה במודלי תרגום:
  podman-compose build --no-cache fixbot
