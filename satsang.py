import os
import queue
import re
import threading
import time
import logging
import numpy as np
import pyaudio
import openai

from faster_whisper import WhisperModel
from flask import Flask, render_template_string
from flask_sock import Sock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("satsang_log.txt", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─── Konfigurace ──────────────────────────────────────────────────────────────
WHISPER_MODEL     = "large-v3"
SAMPLE_RATE       = 16000
CHUNK_SIZE        = 1024
SILENCE_RMS       = 100
SILENCE_CHUNKS    = 25   # počet tichých chunků → konec věty (~1.6 s @ 1024/16 kHz)
MIN_SPEECH_CHUNKS = 20   # minimální délka promluvy (~1.3 s), aby se neposílaly krátké fragmenty
MAX_CHUNKS        = 250  # pojistka: max délka bufferu (~16 s)

# ─── System Prompt ────────────────────────────────────────────────────────────
GLOSSARY = """
संकल्प = Sankalp (záměr, odhodlání)
गौ माता = Gó Mata
गाय माता = Gó Mata
करौली सरकार = Karauli Sarkar
करौली सरकार दरबार = Karauli Darbár
भोलेनाथ = Bholénáth
महाकाल = Mahákál
जय = Sláva
सद्गुरुदेव की जय = Sláva Sadgurudévovi
हर हर महादेव = Har Har Mahádév
हरि हर = Hari Har
दरबार = Darbár
परमात्मा = Paramátma
आत्मा = Átma
कामाख्या = Kamakhya
करौली = Karauli
महाराज = Maharaj
गुरुदेव = Gurudev
शक्ति = Šakti
सिद्धि = Siddhi
बच्चड़ा = Baččadá (telenka)
माया = Májá
धर्म = Dharma
मोक्ष = Mokša
कर्म = Karma
सत्संग = Satsang
साधना = Sádhana
भक्ति = Bhakti
समाधि = Samádhi
मंत्र = Mantra
सेवा = Sevá
दीक्षा = Díkšá
महाकाली = Mahákálí
शंकर = Šankar
महादेव = Mahádév
इष्ट = Išta / Ištadév
इष्टदेव = Ištadév
आराध्य = uctívané Božství
प्रार्थना = modlitba
निवेदन = prosba, předložení prosby
दुःख = utrpení
कष्ट = obtíž, trápení
मुक्ति = osvobození
ध्यान साधना = dhján-sádhana
प्रारब्ध = Prárabdha
पुण्य = Púnja
पाप = Páp
भाग्य = Bhágja (osud)
""".strip()

SYSTEM_PROMPT = f"""Jsi tlumočník satsangů. Tvým úkolem je vytvořit co nejvěrnější český překlad hindského projevu při zachování významu, argumentace a struktury sdělení.

SLOVNÍK TERMÍNŮ:
{GLOSSARY}

PRAVIDLA:

1. Zachovej věrný význam slov mluvčího. Nic nepřidávej, nic nevynechávej.
2. Zachovej všechny příklady, argumenty, opakování, metafory a logickou strukturu sdělení. Neshrnuj. Nezkracuj. Nezobecňuj.
3. Překládej především význam, ale pouze v rozsahu nutném pro přirozenou češtinu. Neměň strukturu sdělení a nevynechávej části textu jen proto, že se zdají opakující.
4. Pokud dostaneš fragment, který je zjevně nedokončený, využij předchozí kontext pouze ke správnému pochopení věty. Nikdy nedoplňuj nové informace ani nové myšlenky.
5. Pokud si nejsi jistý významem části textu, zachovej původní termín v transliteraci namísto odhadu.
6. Zachovej tón a styl mluvčího. Nepřetvářej text do stylu duchovní literatury ani jej stylisticky nevylepšuj.
7. Ponech klíčové termíny: Darbár, Paramátma, Átma, dharma, mokša, karma, satsang, sadhana, bhakti, Gurudev, Maharaj, ji, mantra, samádhi, čit, ánanda, šakti, máyá, líla, sevá, guru, šišja.
8. NEPIŠ komentáře, vysvětlení ani poznámky. Vrať pouze překlad.
9. Pokud je text nesrozumitelný nebo neúplný, přelož pouze část, kterou lze s vysokou jistotou určit.
10. Pokud se v textu objeví výraz, který může být duchovní termín, tradiční zvolání nebo vlastní jméno, upřednostni transliteraci před kreativním překladem.
11. Pokud text působí nedokončeně, nevyvozuj závěry. Přelož pouze to, co je explicitně řečeno.
12. Nikdy nevytvářej etymologie, výklady sanskrtských slov ani duchovní interpretace, pokud nejsou výslovně řečeny mluvčím.

PŘÍKLAD:
Hindština: दरबार आपकी आस्था विश्वास और सरद्धा के अनुसार आपके प्रार्थनाओं को निवेदन को पूर्ण करने की कोशिश करता है
Dobrý překlad: Darbár se snaží naplnit vaše modlitby a prosby — podle vaší víry, důvěry a oddanosti.
Špatný překlad: Darbár je v souladu s vaší vírou, důvěrou a oddaností. Modlitba se snaží naplnit své přání."""

# ─── Globální stav ────────────────────────────────────────────────────────────
audio_queue       = queue.Queue()
translation_queue = queue.Queue()
connected_clients = set()
app  = Flask(__name__)
sock = Sock(app)

# ─── HTML titulky ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="cs">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Satsang · Živý překlad</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0a0a;
    color: #f0ebe1;
    font-family: 'Georgia', serif;
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0 2rem 2rem;
  }
  #hlavicka {
    width: 100%;
    text-align: center;
    flex-shrink: 0;
    padding: 1rem 0 0.5rem;
    font-size: 0.75rem;
    letter-spacing: 0.25em;
    color: #b8922a;
    text-transform: uppercase;
    opacity: 0.8;
  }
  #status {
    position: fixed;
    top: 1rem;
    right: 1.5rem;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: #555;
  }
  #status.live { color: #b8922a; }
  #titulky {
    max-width: 900px;
    width: 100%;
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding-bottom: 1rem;
    scrollbar-width: none;
    position: relative;
  }
  #titulky::-webkit-scrollbar { display: none; }
  #titulky::before {
    content: '';
    position: sticky;
    top: 0;
    left: 0;
    right: 0;
    height: 6rem;
    background: linear-gradient(to bottom, #0a0a0a 0%, transparent 100%);
    flex-shrink: 0;
    pointer-events: none;
    z-index: 1;
  }
  #spacer { flex: 1; }
  .segment {
    font-size: clamp(1.4rem, 3.5vw, 2.2rem);
    line-height: 1.6;
    text-align: center;
    padding: 0.5rem 0;
    flex-shrink: 0;
    opacity: 0;
    animation: fade-in 0.6s ease forwards;
  }
  .segment.aktivni { color: #f7f3ec; }
  .segment.stary    { color: #3a3a3a; }
  #scroll-hint {
    position: fixed;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #b8922a;
    opacity: 0;
    transition: opacity 0.4s;
    pointer-events: none;
    text-transform: uppercase;
  }
  #scroll-hint.visible { opacity: 0.7; }
  @keyframes fade-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  #cekam {
    font-size: 1rem;
    color: #333;
    letter-spacing: 0.15em;
    margin-bottom: 3rem;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50%       { opacity: 0.8; }
  }
</style>
</head>
<body>
<div id="hlavicka">Satsang · Živý překlad</div>
<div id="status">připojování...</div>
<div id="titulky"><div id="spacer"></div><div id="cekam">· · ·</div></div>
<div id="scroll-hint">↑ historie</div>
<script>
  const titulkyEl  = document.getElementById('titulky');
  const statusEl   = document.getElementById('status');
  const hintEl     = document.getElementById('scroll-hint');
  let cekamEl      = document.getElementById('cekam');
  const MAX        = 40;
  let segmenty     = [];

  function isAtBottom() {
    return titulkyEl.scrollHeight - titulkyEl.scrollTop - titulkyEl.clientHeight < 60;
  }

  titulkyEl.addEventListener('scroll', () => {
    hintEl.classList.toggle('visible', !isAtBottom());
  });

  hintEl.style.pointerEvents = 'auto';
  hintEl.style.cursor = 'pointer';
  hintEl.addEventListener('click', () => {
    titulkyEl.scrollTo({ top: titulkyEl.scrollHeight, behavior: 'smooth' });
  });

  const ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onopen  = () => { statusEl.textContent = '● živě'; statusEl.className = 'live'; };
  ws.onclose = () => { statusEl.textContent = 'odpojeno'; statusEl.className = ''; };
  ws.onmessage = (e) => {
    const text = e.data.trim();
    if (!text) return;
    if (cekamEl) { cekamEl.remove(); cekamEl = null; }
    const atBottom = isAtBottom();
    segmenty.push(text);
    if (segmenty.length > MAX) segmenty.shift();
    titulkyEl.innerHTML = '<div id="spacer"></div>';
    segmenty.forEach((s, i) => {
      const div = document.createElement('div');
      div.className = 'segment ' + (i === segmenty.length - 1 ? 'aktivni' : 'stary');
      div.textContent = s;
      titulkyEl.appendChild(div);
    });
    if (atBottom) titulkyEl.scrollTop = titulkyEl.scrollHeight;
  };
</script>
</body>
</html>"""

# ─── Flask ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@sock.route('/ws')
def websocket(ws):
    connected_clients.add(ws)
    try:
        while True:
            time.sleep(1)
    except Exception:
        connected_clients.discard(ws)

def broadcast(text):
    dead = set()
    for client in connected_clients:
        try:
            client.send(text)
        except Exception:
            dead.add(client)
    connected_clients.difference_update(dead)

# ─── Výběr audio zařízení ─────────────────────────────────────────────────────
def najdi_audio_zarizeni(force_index=None):
    pa = pyaudio.PyAudio()
    log.info("📋  Dostupná audio zařízení:")
    blackhole_index = None
    mic_index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            nazev = info['name']
            log.info(f"  [{i}] {nazev}")
            if 'BlackHole' in nazev and blackhole_index is None:
                blackhole_index = i
            if any(k in nazev for k in ('Built-in', 'MacBook', 'Microphone')):
                mic_index = i
    pa.terminate()
    if force_index is not None:
        log.info(f"✅  Vybrané zařízení [{force_index}]")
        return force_index
    if blackhole_index is not None:
        log.info(f"✅  BlackHole [{blackhole_index}] — systémový zvuk")
        return blackhole_index
    elif mic_index is not None:
        log.info(f"✅  Vestavěný mikrofon [{mic_index}]")
        return mic_index
    else:
        log.info("✅  Výchozí zařízení [0]")
        return 0

# ─── Audio thread ─────────────────────────────────────────────────────────────
def audio_thread(force_index=None):
    device_index = najdi_audio_zarizeni(force_index)
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK_SIZE,
    )
    log.info("🎙  Poslouchám...\n")
    buffer = []
    is_speaking = False
    silence_counter = 0

    def flush_buffer():
        if len(buffer) >= MIN_SPEECH_CHUNKS:
            audio_queue.put(b"".join(buffer))
        else:
            print(".", end="", flush=True)
        buffer.clear()

    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio_np**2))

        if rms > SILENCE_RMS:
            is_speaking = True
            silence_counter = 0
            buffer.append(data)
        elif is_speaking:
            buffer.append(data)
            silence_counter += 1
            if silence_counter >= SILENCE_CHUNKS:
                flush_buffer()
                is_speaking = False
                silence_counter = 0

        # pojistka: příliš dlouhý blok (Gurudev mluví nepřetržitě)
        if len(buffer) >= MAX_CHUNKS:
            flush_buffer()
            is_speaking = False
            silence_counter = 0

# ─── Whisper thread ───────────────────────────────────────────────────────────
def whisper_thread():
    log.info("⏳  Načítám Whisper...")
    model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="auto")
    log.info("✅  Whisper připraven.")
    while True:
        raw = audio_queue.get()
        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(
            audio_np,
            language="hi",
            beam_size=8,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt="सत्संग गुरुदेव दरबार साधना भक्ति शक्ति परमात्मा समाधि संकल्प माया दीक्षा सेवा गौ माता गाय बच्चड़ा महाकाली कामाख्या करौली शंकर महादेव भोलेनाथ निवेदन कृतज्ञता ध्यान साधना प्रार्थना समय क्षण पुण्य प्रारब्ध अभ्यास",
            condition_on_previous_text=True,
        )
        text = " ".join(s.text for s in segments).strip()
        if text:
            log.info(f"🇮🇳  {text}")
            translation_queue.put(text)

# ─── OpenAI thread s retry ───────────────────────────────────────────────────
CONTEXT_WINDOW = 20

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def preloz(client, hindi_text, context_history):
    if context_history:
        context = "\n".join(context_history[-CONTEXT_WINDOW:])
        user_content = f"PŘEDCHOZÍ KONTEXT:\n{context}\n\nNOVÝ TEXT K PŘEKLADU:\n{hindi_text}"
    else:
        user_content = hindi_text
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip()

PREFERRED_SIZE   = 300   # ideální velikost bloku pro překlad
MAX_SIZE         = 600   # tvrdý limit — přelož okamžitě
MAX_WAIT         = 45    # max sekund od posledního flushe
SILENCE_FLUSH    = 20    # sekund ticha od posledního segmentu → flush i kratšího bufferu
MIN_FLUSH_SIZE   = 100   # minimum znaků, aby se vůbec posílalo k překladu

def claude_thread():
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    log.info("✅  OpenAI API připojeno.")
    segment_buffer = []
    context_history = []
    last_flush_time = time.time()
    last_segment_time = time.time()

    def do_flush():
        nonlocal last_flush_time
        spojeny_text = " ".join(segment_buffer)
        if len(spojeny_text) < MIN_FLUSH_SIZE:
            return False
        try:
            czech_text = preloz(client, spojeny_text, context_history)
            zakazana = ["potřebuji více kontextu", "prosím poskytnout", "čekám na další"]
            if any(z in czech_text.lower() for z in zakazana):
                return
            czech_text = re.sub(r'[ऀ-ॿ]', '', czech_text).strip()
            if czech_text and czech_text != '""':
                log.info(f"🇨🇿  {czech_text}\n")
                broadcast(czech_text)
                context_history.append(f"HINDI:\n{spojeny_text}\n\nČESKY:\n{czech_text}")
                if len(context_history) > CONTEXT_WINDOW:
                    context_history.pop(0)
        except Exception as e:
            log.error(f"⚠️  Překlad selhal: {e}")
        segment_buffer.clear()
        last_flush_time = time.time()
        return True

    while True:
        try:
            hindi_text = translation_queue.get(timeout=1.0)
        except queue.Empty:
            # nic nepřišlo — zkontroluj, jestli je ticho a buffer neprázdný
            if segment_buffer and (time.time() - last_segment_time) >= SILENCE_FLUSH:
                spojeny = " ".join(segment_buffer)
                log.info(f"🔇  Ticho {SILENCE_FLUSH}s → flush ({len(spojeny)} zn.)")
                if not do_flush():
                    # buffer je příliš krátký, nech ho — ale neopakuj dokud nepřijde nový segment
                    log.info(f"⏳ Buffer příliš krátký ({len(spojeny)} zn.), čekám na další segment")
                    last_segment_time = time.time()  # reset, aby se netriggrovalo znovu
            continue

        segment_buffer.append(hindi_text)
        last_segment_time = time.time()
        spojeny_text = " ".join(segment_buffer)
        text_len = len(spojeny_text)
        elapsed = time.time() - last_flush_time

        too_much = text_len >= MAX_SIZE
        enough_and_waited = text_len >= PREFERRED_SIZE and elapsed >= MAX_WAIT

        if too_much:
            log.info(f"📤  Max size → flush ({text_len} zn.)")
            do_flush()
        elif enough_and_waited:
            log.info(f"📤  Enough + waited {elapsed:.0f}s → flush ({text_len} zn.)")
            do_flush()
        else:
            log.info(f"⏳ Buffer {text_len}/{PREFERRED_SIZE}-{MAX_SIZE} zn., {elapsed:.0f}s")

# ─── Start ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None, help="Index audio zařízení")
    args = parser.parse_args()

    threading.Thread(target=audio_thread,   args=(args.device,), daemon=True).start()
    threading.Thread(target=whisper_thread, daemon=True).start()
    threading.Thread(target=claude_thread,  daemon=True).start()

    log.info("\n🕉  Satsang Překladač")
    log.info("━" * 40)
    log.info("📺  http://localhost:5001")
    log.info("📱  IP zjistíš: ipconfig getifaddr en0")
    log.info("📄  Log: satsang_log.txt")
    log.info("━" * 40 + "\n")

    app.run(host="0.0.0.0", port=5001, debug=False)